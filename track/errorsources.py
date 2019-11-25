"""error sources for use in telescope tracking control loop.

The errorsources module provides classes that derive from the abstract ErrorSource class. Each
class defines a method compute_error() which returns an error vector representing the difference
between the current position and some measure of the ideal position. This information might come
from ephemeris data for a celestial or man-made object, a camera, or even human input. Error source
classes can also be designed to compute an error vector by combining data from multiple sources
or by intelligently switching between sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
import numpy as np
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, Longitude
from astropy.time import Time
try:
    import cv2
    from track import webcam
except ImportError as e:
    if 'cv2' in str(e):
        print('Failed to import cv2. Optical tracking requires OpenCV.')
    raise
from track.mathutils import angle_between, camera_eq_error
from track.telem import TelemSource
from track.mounts import MeridianSide


# pylint: disable=too-few-public-methods
class ErrorSource(ABC):
    """Abstract base class for error sources.

    This class provides some abstract methods to provide a common interface for error sources to be
    used in the tracking loop. All error sources must inheret from this class and implement the
    methods defined.
    """

    class NoSignalException(Exception):
        """Raised when no signal is available for error calculation."""


    @abstractmethod
    def compute_error(self):
        """Computes the error signal.

        Returns:
            PointingError: This contains error terms for each mount axis.

        Raises:
            NoSignalException: If the error cannot be computed.
        """


# pylint: disable=too-few-public-methods
class PointingError(NamedTuple):
    """Set of pointing error terms.

    This contains error terms for telescope mounts having two motorized axes, which should cover
    the vast majority of amateur mounts. This code is intended for use with equatorial mounts,
    altitude-azimuth mounts, and equatorial mounts where the polar axis is intentionally not
    aligned with the celesital pole. Therefore the members of this tuple do not use the
    conventional axis names such as "ra" for "right-ascension" or "dec" for "declination" since
    these names are not appropriate in all scenarios.

    Both attirubutes are instances of the Astropy Angle class. The range of values should be
    restricted to [-360, +360] degrees.

    Attributes:
        encoder_0: The axis closer to the base of the mount. For an equatorial mount this is
            usually the right ascension axis. For az-alt mounts this is usually the azimuth axis.
        encoder_1: The axis further from the base of the mount but closer to the optical tube. For
            an equatorial mount this is usually the declination axis. For az-alt mounts this is
            usually the altitude axis.
    """
    encoder_0: Angle
    encoder_1: Angle


class BlindErrorSource(ErrorSource, TelemSource):
    """Ephemeris based error source.

    This class implements an error source based on computed ephemeris information. It is dubbed
    a "blind" error source in the sense that no physical sensing of the target position is
    involved (such as by use of a camera).

    Attributes:
        meridian_side (MeridianSide): Allows selection of a meridian side for equatorial mounts.
        mount (TelescopeMount): Provides a generic interface to a telescope mount.
        mount_model (MountModel): Transforms from sky coordinates to mount encoder positions.
        observer (PyEphem Observer): PyEphem object representing the same coordinates as the
            location attribute.
        target (PyEphem Target): Represents the target object of interest.
        target_position_offset (PositionOffset): Offset to be applied to the target's position each
            time it is calculated from the ephemeris. Set to None by default, in which case the
            position is not adjusted.
        _telem_channels (dict): A collection of telemetry channel values cached such that they can
            be read by the telemetry polling thread. This is updated on each call to
            compute_error().
    """


    class PositionOffset(NamedTuple):
        """Specifies an offset to be applied to a raw target position predicted from ephemeris.

        Attributes:
            direction: An angle that specifies the direction in which to apply the offset. A
                direction angle of 0 is the direction of the target's current motion, and thus a
                positive separation will move the target position further ahead in the trajectory.
            separation: An angle that specifies the magnitude of the offset to apply. This is the
                on-sky separation between the raw target and the offset target.
        """
        direction: Angle
        separation: Angle


    def __init__(
            self,
            mount,
            mount_model,
            target,
            meridian_side=MeridianSide.WEST
        ):
        """Inits BlindErrorSource object.

        Args:
            mount (TelescopeMount): Error terms will be computed relative to the encoder positions
                reported by this mount.
            target (Target): Object that identifies the target to be tracked.
            meridian_side (MeridianSide): For equatorial mounts indicates which side of the
                meridian is desired.
        """
        self.target = target
        self.mount = mount
        self.mount_model = mount_model
        self.meridian_side = meridian_side
        self.target_position_offset = None
        self._telem_channels = {}

        # Create a PyEphem Observer object
        self.observer = ephem.Observer()
        self.observer.lat = mount_model.location.lat.deg
        self.observer.lon = mount_model.location.lon.deg
        self.observer.elevation = mount_model.location.height.to_value(u.m)


    def _compute_target_position(self, when):
        """Get the position of the target at a specific time for the observer location.

        Args:
            when (datetime): Time of observation.

        Returns:
            SkyCoord: Coordinates of the target from the observer's location.
        """
        self.observer.date = ephem.Date(when)
        self.target.compute(self.observer)
        return SkyCoord(self.target.ra * u.rad, self.target.dec * u.rad)


    def _offset_target_position(self, position, offset):
        """Offset the target position by a specified amount in a trajectory-relative direction.

        This shifts the position of the target. This could be used to adjust for pointing errors
        based on real-time observations of the target. The adjustments are made relative to the
        current apparent motion vector of the target such that it is easier to move it forward,
        backward, or side-to-side along its current trajectory. This is much more intuitive than
        trying to figure out the same adjustment in some coordinate system.

        Args:
            position (SkyCoord): The target position to be adjusted.
            offset (PositionOffset): Offset to be applied.
        """

        # estimate direction vector of the target using positions now and several seconds later
        now = datetime.now(timezone.utc)
        position_now = self._compute_target_position(now)
        position_later = self._compute_target_position(now + timedelta(seconds=10))

        # position angle pointing in direction of target motion
        trajectory_position_angle = position_now.position_angle(position_later)

        # position angle in direction of offset to be applied
        offset_position_angle = trajectory_position_angle + offset.direction

        return position.directional_offset_by(offset_position_angle, offset.separation)


    @staticmethod
    def _compute_axis_error(mount_enc_position, target_enc_position, no_cross_position):
        """Compute error term for a single axis taking into account no-cross positions

        The shortest path on an axis is just the difference between the target and mount encoder
        positions wrapped to [-180, +180] degrees. However the shortest path is not always allowed
        because it may result in cord wrap or send the mount moving towards its built-in axis
        limits. This function contains the logic needed to account for this, resulting in error
        terms that are either the shortest distance if allowed or the longer distance if
        constraints prevent taking the short path.

        Args:
            mount_enc_position (Longitude): The mount encoder position.
            target_enc_position (Longitude): The target encoder position.
            no_cross_position (Longitude): An encoder position that this axis is not allowed to
                cross.

        Returns:
            Longitude: The error term for this axis, with wrap_angle set to 180 degrees.
        """

        # shortest path
        prelim_error = Angle(
            Longitude(mount_enc_position - target_enc_position, wrap_angle=180*u.deg)
        )

        # No limit, no problem! In this case always take the shortest distance path.
        if no_cross_position is None:
            return prelim_error

        # error term as if the no-cross point were the target position
        no_cross_error = Angle(
            Longitude(mount_enc_position - no_cross_position, wrap_angle=180*u.deg)
        )

        # actual target is closer than the no-cross point so can't possibly be crossing it
        if abs(prelim_error) <= abs(no_cross_error):
            return prelim_error

        # actual target is in opposite direction from the no-cross point
        if np.sign(prelim_error) != np.sign(no_cross_error):
            return prelim_error

        # switch direction since prelim_error would have crossed the no-cross point
        return prelim_error + 360*u.deg if prelim_error < 0 else prelim_error - 360*u.deg


    def compute_error(self):
        """Compute encoder error terms based on target ephemeris.

        The error terms for each mount axis are found by the following procedure (some details
        are omitted from this outline for simplicity):

        1) Calculate the position of the target in local astrometric equatorial coordinates for the
           observer's location and for the current time.
        2) Use the mount model to transform the target coordinates to mount encoder positions.
        3) Subtract the current encoder positions from the target encoder positions.

        Returns:
            PositionError: Error terms for each mount axis.
        """

        datetime_now = datetime.now(timezone.utc)
        astropy_time = Time(datetime_now)

        # get coordinates of target for current time
        target_position_raw = self._compute_target_position(datetime_now)

        if self.target_position_offset is not None:
            target_position = self._offset_target_position(
                target_position_raw,
                self.target_position_offset
            )
        else:
            target_position = target_position_raw

        # transform from astrometric coordinate system to mount encoder positions
        target_enc_positions = self.mount_model.world_to_mount(
            target_position,
            self.meridian_side,
            astropy_time
        )

        # get current position of telescope encoders
        mount_enc_positions = self.mount.get_position()

        pointing_error = PointingError(
            *[self._compute_axis_error(
                mount_enc_positions[idx],
                target_enc_positions[idx],
                self.mount.no_cross_encoder_positions()[idx],
            ) for idx in range(2)]
        )

        # TODO: Update self._telem_channels dict. Use a mutex. See control.py for example.

        return pointing_error


    def get_telem_channels(self):
        return self._telem_channels


class OpticalErrorSource(ErrorSource, TelemSource):
    """Computer vision based error source.

    This class implements an error source based on computer vision recognition of a target in an
    image from a camera. The error vector of the detected target from image center is transformed
    to an error vector in the mount's coordinate system.

    Attributes:
        degrees_per_pixel: Apparent size of a photosite in degrees.
        webcam: A WebCam object instance.
        x_axis_name: Name of mount axis parallel to the camera's x-axis.
        y_axis_name: Name of mount axis parallel to the camera's y-axis.
        frame_width_px: Width of the image in pixels.
        frame_height_px: Height of the image in pixels.
        frame_center_px: A tuple with the coordinates of the image center.
        concec_detect_frames: Number of consecutive frames where a target was detected.
        consec_no_detect_frames: Number of consecutive frames since a target was last detected.
        detector: An OpenCV SimpleBlobDetector object.
    """

    def __init__(
            self,
            cam_dev_path,
            arcsecs_per_pixel,
            cam_num_buffers,
            cam_ctlval_exposure,
            x_axis_name,
            y_axis_name,
            mount=None,
            frame_dump_dir=None,
        ):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0

        self.webcam = webcam.WebCam(
            cam_dev_path,
            cam_num_buffers,
            cam_ctlval_exposure,
            frame_dump_dir
        )

        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

        self.mount = mount

        self.frame_width_px = self.webcam.get_res_x()
        self.frame_height_px = self.webcam.get_res_y()

        self.frame_center_px = (self.frame_width_px / 2.0, self.frame_height_px / 2.0)

        # cached values from last error calculation (for telemetry)
        self.error_cached = {}

        # counts of consecutive frames with detections or no detections
        self.consec_detect_frames = 0
        self.consec_no_detect_frames = 0

        cv2.namedWindow('frame')

    def get_axis_names(self):
        return [self.x_axis_name, self.y_axis_name]

    def find_features(self, frame):
        """Find bright features in a camera frame.

        Args:
            frame: A camera frame.

        Returns:
            A list of keypoints.
        """

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # pick a threshold as the max of these two methods:
        # - 99th percentile of histogram
        # - peak of histogram plus a magic number
        hist, bins = np.histogram(gray.ravel(), 256, [0,256])
        cumsum = np.cumsum(hist)
        threshold_1 = np.argmax(cumsum >= 0.99*cumsum[-1])
        threshold_2 = np.argmax(hist) + 4
        threshold = max(threshold_1, threshold_2)

        retval, thresh = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        # outer contours only
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        keypoints = []
        for contour in contours:
            moms = cv2.moments(contour);
            if moms['m00'] == 0.0:
                continue

            # find the center of mass
            center = np.array([moms['m10'] / moms['m00'], moms['m01'] / moms['m00']])

            # find the maximum distance between the center and the contour
            radius = 0.0
            for p in contour:
                dist = np.linalg.norm(center - p)
                radius = max(radius, dist)

            keypoints.append(cv2.KeyPoint(center[0], center[1], 2.0*radius))

        return keypoints

    def show_annotated_frame(self, frame, keypoints=[], target_keypoint=None):
        """Displays camera frame in a window with features circled and crosshairs.

        Args:
            frame: Camera frame.
            keypoints: List of all keypoints.
            target_keypoint: The keypoint identified as the target.
        """

        frame_annotated = frame.copy()

        # add grey crosshairs
        cv2.line(
            frame_annotated,
            (int(self.frame_center_px[0]), 0),
            (int(self.frame_center_px[0]), int(self.frame_height_px) - 1),
            (50, 50, 50),
            1
        )
        cv2.line(
            frame_annotated,
            (0, int(self.frame_center_px[1])),
            (int(self.frame_width_px) - 1, int(self.frame_center_px[1])),
            (50, 50, 50),
            1
        )

        # lower bound on keypoint size so the annotation is visible (okay to modify size because
        # we only really care about the center)
        for point in keypoints:
            point.size = max(point.size, 10.0)

        # cicle non-target keypoints in blue
        frame_annotated = cv2.drawKeypoints(
            frame_annotated,
            keypoints,
            np.array([]),
            (255, 0, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # circle target keypoint in red
        if target_keypoint is not None:
            target_keypoint.size = max(target_keypoint.size, 10.0)
            frame_annotated = cv2.drawKeypoints(
                frame_annotated,
                [target_keypoint],
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        # display the frame in a window
        cv2.imshow('frame', frame_annotated)
        cv2.waitKey(1)

    def compute_error(self, blocking=True):

        frame = self.webcam.get_fresh_frame(blocking=blocking)

        if frame is None:
            self.error_cached = {}
            raise self.NoSignalException('No frame was available')

        keypoints = self.find_features(frame)

        if not keypoints:
            self.show_annotated_frame(frame)
            self.consec_detect_frames = 0
            self.consec_no_detect_frames += 1
            self.error_cached = {}
            raise self.NoSignalException('No target identified')

        # use the keypoint closest to the center of the frame
        min_error = None
        target_keypoint = None
        for keypoint in keypoints:
            # error is the vector between the keypoint and center frame
            e_x = self.frame_center_px[0] - keypoint.pt[0]
            e_y = keypoint.pt[1] - self.frame_center_px[1]
            error_mag = np.abs(e_x + 1j*e_y)

            if min_error is None or error_mag < min_error:
                target_keypoint = keypoint
                min_error = error_mag
                error_x_px = e_x
                error_y_px = e_y

        error_x_deg = error_x_px * self.degrees_per_pixel
        error_y_deg = error_y_px * self.degrees_per_pixel

        if self.mount is None:
            # This is a crude approximation. It breaks down near the mount's pole.
            error = {}
            error[self.x_axis_name] = error_x_deg
            error[self.y_axis_name] = error_y_deg
        else:
            # This is the "proper" way of doing things but it requires knowledge of the mount
            # position.
            error = camera_eq_error(
                self.mount.get_position(max_cache_age=0.1),
                [error_x_px, error_y_px],
                self.degrees_per_pixel
            )
        # angular separation between detected target and center of camera frame in degrees
        error['mag'] = np.abs(error_x_deg + 1j*error_y_deg)
        self.error_cached = error

        self.consec_detect_frames += 1
        self.consec_no_detect_frames = 0

        self.show_annotated_frame(frame, keypoints, target_keypoint)

        return error

    def get_telem_channels(self):
        chans = {}
        chans['consec_detect_frames'] = self.consec_detect_frames
        chans['consec_no_detect_frames'] = self.consec_no_detect_frames
        for key, val in self.error_cached.items():
            chans['error_' + key] = val
        return chans


class HybridErrorSource(ErrorSource, TelemSource):
    """Hybrid of blind and computer vision error sources.

    This class is a hybrid of the BlindErrorSource and the OpticalErrorSource classes. It computes
    error vectors using both and uses a simple state machine to select between them. This allows
    for an acquisition phase that relies on blind tracking and then switches to (hopefully) more
    accurate tracking with a camera when the target is detected within the camera's field of view.
    Falls back on blind tracking if the two error sources diverge too much or if the optical error
    source is unable to detect the target for too long. Fall back on blind tracking on divergence
    is based on the assumption that the blind error source is generally more reliable but less
    accurate--for example, the computer vision algorithm can be tricked by false targets.

    Attributes:
        axes: List of mount axis names.
        blind: BlindErrorSource object instance.
        optical: OpticalErrorSource object instance.
        max_divergence: Max allowed divergence between error sources in degrees.
        max_optical_no_signal_frames: Reverts to blind mode after this many frames with no target.
        state: String used as an enum to represent state: 'blind' or 'optical'.
    """
    def __init__(
            self,
            mount,
            observer,
            target,
            cam_dev_path,
            arcsecs_per_pixel,
            cam_num_buffers,
            cam_ctlval_exposure,
            max_divergence=5.0,
            max_optical_no_signal_frames=4,
            meridian_side='west',
            frame_dump_dir=None,
        ):
        self.axes = mount.get_axis_names()
        self.blind = BlindErrorSource(
            mount,
            observer,
            target,
            meridian_side=meridian_side
        )
        # FIXME: Have to do this because OpticalErrorSource has a crappy way of specifying how the
        # camera is oriented with respect to the mount axes.
        if set(self.axes) == set(['az', 'alt']):
            self.optical = OpticalErrorSource(
                cam_dev_path,
                arcsecs_per_pixel,
                cam_num_buffers,
                cam_ctlval_exposure,
                x_axis_name='az',
                y_axis_name='alt',
                frame_dump_dir=frame_dump_dir,
            )
        elif set(self.axes) == set(['ra', 'dec']):
            self.optical = OpticalErrorSource(
                cam_dev_path,
                arcsecs_per_pixel,
                cam_num_buffers,
                cam_ctlval_exposure,
                x_axis_name='ra',
                y_axis_name='dec',
                mount=mount,
                frame_dump_dir=frame_dump_dir,
            )
        else:
            raise ValueError('unrecognized axis names')
        self.max_divergence = max_divergence
        self.max_optical_no_signal_frames = max_optical_no_signal_frames
        self.state = 'blind'
        self.divergence_angle = None
        print('Hybrid error source starting in blind tracking state')

    def get_axis_names(self):
        return self.axes

    def register_blind_offset_callback(self, callback):
        """See BlindErrorSource documentation."""
        self.blind.register_offset_callback(callback)

    def compute_error(self):
        blind_error = self.blind.compute_error()

        try:
            optical_error = self.optical.compute_error(blocking=(self.state == 'optical'))
        except ErrorSource.NoSignalException:
            self.divergence_angle = None
            if self.state == 'blind':
                return blind_error
            else:
                if self.optical.consec_no_detect_frames >= self.max_optical_no_signal_frames:
                    print('Lost target in camera, switching to blind tracking')
                    self.state = 'blind'
                    return blind_error
                raise

        # get optical target position, which is the mount's current position plus the optical
        # error vector
        mount_position = self.blind.mount_position_cached
        target_position_optical = {}
        for axis in self.axes:
            target_position_optical[axis] = mount_position[axis] + optical_error[axis]

        # get blind object position, which is what PyEphem says it is
        target_position_blind = self.blind.target_position_cached

        # Get angle between optical and blind target position solutions. This is a measure of how
        # far the two solutions have diverged. A large divergence angle could mean that the
        # computer vision algorithm is not tracking the correct object.
        self.divergence_angle = angle_between(target_position_optical, target_position_blind)

        if self.state == 'blind' and self.divergence_angle < self.max_divergence:
            print('A target is in view, switching to optical tracking')
            self.state = 'optical'
        elif self.state == 'optical' and self.divergence_angle > self.max_divergence:
            print('Solutions diverged, switching to blind tracking')
            self.state = 'blind'

        return blind_error if self.state == 'blind' else optical_error

    def get_telem_channels(self):
        chans = {}
        chans['state'] = 0 if self.state == 'blind' else 1
        chans['divergence_angle'] = self.divergence_angle
        return chans
