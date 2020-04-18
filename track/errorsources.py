"""error sources for use in telescope tracking control loop.

The errorsources module provides classes that derive from the abstract ErrorSource class. Each
class defines a method compute_error() which returns an error vector representing the difference
between the current position and some measure of the ideal position. This information might come
from ephemeris data for a celestial or man-made object, a camera, or even human input. Error source
classes can also be designed to compute an error vector by combining data from multiple sources
or by intelligently switching between sources.
"""

import threading
from abc import abstractmethod
from enum import IntEnum
from typing import NamedTuple, Optional, Tuple, List, Dict
from math import inf
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, Longitude, UnitSphericalRepresentation
from astropy.time import Time, TimeDelta
try:
    import cv2
except ImportError as e:
    if 'cv2' in str(e):
        print('Failed to import cv2. Optical tracking requires OpenCV.')
    raise
from track.cameras import Camera
from track.model import MountModel
from track.mounts import TelescopeMount, MeridianSide
from track.targets import Target
from track.telem import TelemSource


class PointingError(NamedTuple):
    """Set of pointing error terms.

    This contains error terms for telescope mounts having two motorized axes, which should cover
    the vast majority of amateur mounts. This code is intended for use with equatorial mounts,
    altitude-azimuth mounts, and equatorial mounts where the polar axis is intentionally not
    aligned with the celesital pole. Therefore the members of this tuple do not use the
    conventional axis names such as "ra" for "right-ascension" or "dec" for "declination" since
    these names are not appropriate in all scenarios.

    Attributes:
        encoder_0: The axis closer to the base of the mount. For an equatorial mount this is
            usually the right ascension axis. For az-alt mounts this is usually the azimuth axis.
             Range is [-360, +360] degrees.
        encoder_1: The axis further from the base of the mount but closer to the optical tube. For
            an equatorial mount this is usually the declination axis. For az-alt mounts this is
            usually the altitude axis. Range is [-360, +360] degrees.
        magnitude: The separation angle between the target and the mount's current position on the
            sky. The individual encoder error terms could be significantly larger than this value
            when the target is near the mount's pole. Range is [0, +180] degrees.
    """
    encoder_0: Angle
    encoder_1: Angle
    magnitude: Angle


class ErrorSource(TelemSource):
    """Abstract base class for error sources.

    This class provides some abstract methods to provide a common interface for error sources to be
    used in the tracking loop. All error sources must inheret from this class and implement the
    methods defined.

    Attributes:
        _telem_channels (dict): A collection of telemetry channel values cached such that they can
            be read by the telemetry polling thread in get_telem_channels(). This should be updated
            on each call to compute_error(). All channels should be updated at once while holding
            _telem_mutex.
        _telem_mutex: Accesses to _telem_channels should always be protected by holding this mutex
            since telemetry polling occurs in a separate thread.
    """

    class NoSignalException(Exception):
        """Raised when no signal is available for error calculation."""


    def __init__(self):
        """Super class constructor"""
        self._telem_chans = {}
        self._telem_mutex = threading.Lock()


    @abstractmethod
    def compute_error(self) -> PointingError:
        """Computes the error signal.

        Returns:
            PointingError: This contains error terms for each mount axis.

        Raises:
            NoSignalException: If the error cannot be computed for any reason.
        """

    def get_telem_channels(self):
        """Called by telemetry polling thread -- see TelemSource abstract base class"""
        # Protect dict copy with mutex since this method is called from another thread
        self._telem_mutex.acquire()
        chans = self._telem_chans.copy()
        self._telem_mutex.release()
        return chans

    @staticmethod
    def _smallest_allowed_error(
            mount_enc_position: Longitude,
            target_enc_position: Longitude,
            no_cross_position: Optional[Longitude],
        ) -> Angle:
        """Compute error term for a single axis taking into account no-cross positions

        The shortest path on an axis is just the difference between the target and mount encoder
        positions wrapped to [-180, +180] degrees. However the shortest path is not always allowed
        because it may result in cord wrap or send the mount moving towards its built-in axis
        limits. This function contains the logic needed to account for this, resulting in error
        terms that are either the shortest distance if allowed or the longer distance if
        constraints prevent taking the short path.

        Args:
            mount_enc_position: The mount encoder position.
            target_enc_position: The target encoder position.
            no_cross_position: Encoder position that this axis is not allowed to cross or None if
                no such position exists.

        Returns:
            The error term for this axis, with wrap_angle set to 180 degrees.
        """

        # shortest path
        prelim_error = Angle(
            Longitude(target_enc_position - mount_enc_position, wrap_angle=180*u.deg)
        )

        # No limit, no problem! In this case always take the shortest distance path.
        if no_cross_position is None:
            return prelim_error

        # error term as if the no-cross point were the target position
        no_cross_error = Angle(
            Longitude(no_cross_position - mount_enc_position, wrap_angle=180*u.deg)
        )

        # actual target is closer than the no-cross point so can't possibly be crossing it
        if abs(prelim_error) <= abs(no_cross_error):
            return prelim_error

        # actual target is in opposite direction from the no-cross point
        if np.sign(prelim_error) != np.sign(no_cross_error):
            return prelim_error

        # switch direction since prelim_error would have crossed the no-cross point
        return prelim_error + 360*u.deg if prelim_error < 0 else prelim_error - 360*u.deg


class BlindErrorSource(ErrorSource):
    """Ephemeris based error source.

    This class implements an error source based on computed ephemeris information. It is dubbed
    a "blind" error source in the sense that no physical sensing of the target position is
    involved (such as by use of a camera).

    Attributes:
        meridian_side (MeridianSide): Allows selection of a meridian side for equatorial mounts.
        mount (TelescopeMount): Provides a generic interface to a telescope mount.
        mount_model (MountModel): Transforms from sky coordinates to mount encoder positions.
        target (Target): Represents the target object of interest.
        target_position_offset (PositionOffset): Offset to be applied to the target's position each
            time it is calculated from the ephemeris. Set to None by default, in which case the
            position is not adjusted.
        target_position (SkyCoord): Predicted topocentric position of the target.
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
            mount: TelescopeMount,
            mount_model: MountModel,
            target: Target,
            meridian_side: MeridianSide = MeridianSide.WEST
        ):
        """Inits BlindErrorSource object.

        Args:
            mount: Error terms will be computed relative to the encoder positions reported by this
                mount.
            mount_model: A model that transforms celestial coordinates to mount encoder positions.
            target: Object that identifies the target to be tracked.
            meridian_side: For equatorial mounts indicates which side of the meridian is desired.
        """
        self.target = target
        self.mount = mount
        self.mount_model = mount_model
        self.meridian_side = meridian_side
        self.target_position_offset = None
        self.target_position = None
        super().__init__()


    def _offset_target_position(self, position: SkyCoord, offset: PositionOffset) -> SkyCoord:
        """Offset the target position by a specified amount in a trajectory-relative direction.

        This shifts the position of the target. This could be used to adjust for pointing errors
        based on real-time observations of the target. The adjustments are made relative to the
        current apparent motion vector of the target such that it is easier to move it forward,
        backward, or side-to-side along its current trajectory. This is much more intuitive than
        trying to figure out the same adjustment in some coordinate system.

        Args:
            position: The target position to be adjusted.
            offset: Offset to be applied.

        Returns:
            Target position with offset applied.
        """

        # estimate direction vector of the target using positions now and several seconds later
        now = Time.now()
        position_now = self.target.get_position(now)
        position_later = self.target.get_position(now + TimeDelta(10.0, format='sec'))

        # position angle pointing in direction of target motion
        trajectory_position_angle = position_now.position_angle(position_later)

        # position angle in direction of offset to be applied
        offset_position_angle = trajectory_position_angle + offset.direction

        return position.directional_offset_by(offset_position_angle, offset.separation)


    def compute_error(self) -> PointingError:
        """Compute encoder error terms based on target ephemeris.

        The error terms for each mount axis are found by the following procedure (some details
        are omitted from this outline for simplicity):

        1) Calculate the position of the target in topocentric coordinates for the observer's
           location and for the current time.
        2) Use the mount model to transform the target coordinates to mount encoder positions.
        3) Subtract the current encoder positions from the target encoder positions.

        Returns:
            Error terms for each mount axis.
        """

        # get current position of mount encoders
        mount_enc_positions = self.mount.get_position()

        # Get topocentric coordinates of target for current time. Critical that the time passed to
        # get_position() is measured as close as possible to when the mount encoder positions were
        # queried.
        target_position_raw = self.target.get_position(Time.now())

        if self.target_position_offset is not None:
            target_position = self._offset_target_position(
                target_position_raw,
                self.target_position_offset
            )
        else:
            target_position = target_position_raw

        # transform from topocentric coordinate system to mount encoder positions
        target_enc_positions = self.mount_model.topocentric_to_encoders(
            target_position,
            self.meridian_side,
        )

        # required for error magnitude calcaulation
        mount_topocentric = self.mount_model.encoders_to_topocentric(mount_enc_positions)
        error_magnitude = mount_topocentric.separation(target_position)

        pointing_error = PointingError(
            *[self._smallest_allowed_error(
                mount_enc_positions[axis],
                target_enc_positions[axis],
                self.mount.no_cross_encoder_positions()[axis],
            ) for axis in self.mount.AxisName],
            error_magnitude
        )

        # update telemetry channels dict in one atomic operation
        self._telem_mutex.acquire()
        self._telem_chans = {}
        self._telem_chans['target_raw_az'] = target_position_raw.az.deg
        self._telem_chans['target_raw_alt'] = target_position_raw.alt.deg
        self._telem_chans['target_az'] = target_position.az.deg
        self._telem_chans['target_alt'] = target_position.alt.deg
        if self.target_position_offset is not None:
            self._telem_chans['target_offset_dir'] = self.target_position_offset.direction.deg
            self._telem_chans['target_offset_sep'] = self.target_position_offset.separation.deg
        self._telem_chans['mount_az'] = mount_topocentric.az.deg
        self._telem_chans['mount_alt'] = mount_topocentric.alt.deg
        self._telem_chans['error_mag'] = error_magnitude.deg
        for axis in self.mount.AxisName:
            self._telem_chans[f'target_enc_{axis}'] = target_enc_positions[axis].deg
            self._telem_chans[f'mount_enc_{axis}'] = mount_enc_positions[axis].deg
            self._telem_chans[f'error_enc_{axis}'] = pointing_error[axis].deg
        self._telem_mutex.release()

        # for use by HybridErrorSource
        self.target_position = target_position

        return pointing_error


class OpticalErrorSource(ErrorSource):
    """Computer vision based error source.

    This class implements an error source based on computer vision recognition of a target in an
    image from a camera. The error vector of the detected target from image center is transformed
    to error terms for each mount axis.

    Attributes:
        camera: An instance of class Camera.
        mount: An instance of class TelescopeMount.
        mount_model: An instance of class MountModel.
        target_position: SkyCoord with AltAz frame giving estimated topocentric position of target.
            Updated on each call to `compute_error()`.
        meridian_side: An instance of MeridianSide.
        frame_center_px: A tuple with the coordinates of the image center.
        concec_detect_frames: Number of consecutive frames where a target was detected.
        consec_no_detect_frames: Number of consecutive frames since a target was last detected.
    """

    def __init__(
            self,
            camera: Camera,
            mount: TelescopeMount,
            mount_model: MountModel,
            meridian_side: Optional[MeridianSide] = None,
            camera_timeout: float = inf,
        ):
        """Construct an instance of OpticalErrorSource

        Args:
            camera: Camera from which to capture imagery.
            mount: Required so current position can be queried.
            mount_model: Required to transform between camera and mount encoder coordinates.
            meridian_side: Mount will stay on this side of the meridian. If None, the mount will
                remain on the same side of the meridian that it is on when this constructor is
                invoked.
            camera_timeout: How long to wait for a frame from the camera in seconds on calls to
                `compute_error()`. If `inf`, `compute_error()` will block indefinitely.
        """
        self.camera = camera
        self.camera_timeout = camera_timeout
        self.mount = mount
        self.mount_model = mount_model
        self.target_position = None

        if meridian_side is not None:
            self.meridian_side = meridian_side
        else:
            _, self.meridian_side = mount_model.encoder_to_spherical(mount.get_position())

        frame_height, frame_width = camera.frame_shape
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)

        # counts of consecutive frames with detections or no detections
        self.consec_detect_frames = 0
        self.consec_no_detect_frames = 0

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', frame_width, frame_height)

        super().__init__()

    @staticmethod
    def find_features(frame: np.ndarray) -> List[cv2.KeyPoint]:
        """Find bright features in a camera frame.

        Args:
            frame: A camera frame.

        Returns:
            A list of keypoints.
        """

        # pick a threshold as the max of these two methods:
        # - 99th percentile of histogram
        # - peak of histogram plus a magic number
        hist, _ = np.histogram(frame.ravel(), 256, [0, 256])
        cumsum = np.cumsum(hist)
        threshold_1 = np.argmax(cumsum >= 0.99*cumsum[-1])
        threshold_2 = np.argmax(hist) + 4
        threshold = max(threshold_1, threshold_2)

        _, thresh = cv2.threshold(
            frame,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        # outer contours only
        _, contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        keypoints = []
        for contour in contours:
            moms = cv2.moments(contour)
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

    def show_annotated_frame(
            self,
            frame: np.ndarray,
            keypoints: Optional[List[cv2.KeyPoint]] = None,
            target_keypoint: Optional[cv2.KeyPoint] = None
        ) -> None:
        """Displays camera frame in a window with features circled and crosshairs.

        Args:
            frame: Camera frame.
            keypoints: List of all keypoints.
            target_keypoint: The keypoint identified as the target.
        """

        frame_annotated = frame.copy()

        # add grey crosshairs
        frame_height, frame_width = self.camera.frame_shape
        cv2.line(
            frame_annotated,
            (int(self.frame_center_px[0]), 0),
            (int(self.frame_center_px[0]), frame_height - 1),
            (50, 50, 50),
            1
        )
        cv2.line(
            frame_annotated,
            (0, int(self.frame_center_px[1])),
            (frame_width - 1, int(self.frame_center_px[1])),
            (50, 50, 50),
            1
        )

        if keypoints is not None:
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


    def _camera_to_mount_error(self, error_x: Angle, error_y: Angle) -> Tuple[PointingError, Dict]:
        """Transform from error terms in camera frame to mount encoder error terms

        Args:
            error_x: Error term in camera's x-axis
            error_y: Error term in camera's y-axis

        Returns:
            Pointing error and dict of telemetry channels.
        """

        # angular separation and direction from center of camera frame to target
        error_complex = error_x.deg + 1j*error_y.deg
        error_magnitude = Angle(np.abs(error_complex) * u.deg)
        error_direction = Angle(np.angle(error_complex) * u.rad)

        # Current position of mount is assumed to be the position of the center of the camera frame
        mount_enc_positions = self.mount.get_position(max_cache_age=0.1)
        mount_coord, mount_meridian_side = self.mount_model.encoder_to_spherical(
            mount_enc_positions
        )

        # Find the position of the target relative to the mount position
        target_position_angle = error_direction + self.mount_model.guide_cam_orientation
        if mount_meridian_side == MeridianSide.EAST:
            # camera orientation flips when crossing the pole
            target_position_angle += 180*u.deg
        target_coord = SkyCoord(mount_coord).directional_offset_by(
            position_angle=target_position_angle,
            separation=error_magnitude
        ).represent_as(UnitSphericalRepresentation)

        # convert target position back to mount encoder positions
        target_enc_positions = self.mount_model.spherical_to_encoder(
            coord=target_coord,
            meridian_side=self.meridian_side,
        )

        # find the resulting mount encoder error terms
        pointing_error = PointingError(
            *[self._smallest_allowed_error(
                mount_enc_positions[axis],
                target_enc_positions[axis],
                self.mount.no_cross_encoder_positions()[axis],
            ) for axis in self.mount.AxisName],
            error_magnitude
        )

        # find target position in topocentric frame (used by HybridErrorSource)
        self.target_position = self.mount_model.encoders_to_topocentric(target_enc_positions)

        telem_chans = {}
        telem_chans['error_mag'] = error_magnitude.deg
        telem_chans['error_cam_direction'] = error_direction.deg
        telem_chans['target_position_angle'] = target_position_angle.deg
        for axis in self.mount.AxisName:
            telem_chans[f'target_enc_{axis}'] = target_enc_positions[axis].deg
            telem_chans[f'mount_enc_{axis}'] = mount_enc_positions[axis].deg
            telem_chans[f'error_enc_{axis}'] = pointing_error[axis].deg

        return pointing_error, telem_chans


    def compute_error(self) -> PointingError:
        """Compute pointing error using computer vision and a camera.

        The encoder error terms are computed using this rough procedure:
        1) A frame is obtained from the camera
        2) Computer vision algorithms are applied to identify bright objects
        3) The bright object nearest the center of the camera frame is assumed to be the target
        4) The error vector for the selected object is computed in the camera's frame
        5) The error vector is transformed from camera frame to mount encoder error terms

        Returns:
            Error terms for each mount axis.
        """

        frame = self.camera.get_frame(timeout=self.camera_timeout)

        if frame is None:
            self._set_telem_channels()
            raise self.NoSignalException('No frame was available')

        keypoints = self.find_features(frame)

        if not keypoints:
            self.consec_detect_frames = 0
            self.consec_no_detect_frames += 1
            self.show_annotated_frame(frame)
            self._set_telem_channels()
            raise self.NoSignalException('No target identified')

        self.consec_detect_frames += 1
        self.consec_no_detect_frames = 0

        # find the keypoint closest to the center of the frame
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

        self.show_annotated_frame(frame, keypoints, target_keypoint)

        # error terms in camera frame
        error_x = Angle(error_x_px * self.camera.pixel_scale * u.deg)
        error_y = Angle(error_y_px * self.camera.pixel_scale * u.deg)

        # transform to mount encoder error terms
        pointing_error, telem = self._camera_to_mount_error(error_x, error_y)

        telem['error_cam_x'] = error_x.deg
        telem['error_cam_y'] = error_y.deg

        self._set_telem_channels(telem)

        return pointing_error

    def _set_telem_channels(self, chans: Optional[Dict] = None) -> None:
        """Set telemetry dict polled by telemetry thread"""
        self._telem_mutex.acquire()
        self._telem_chans = {}
        self._telem_chans['consec_detect_frames'] = self.consec_detect_frames
        self._telem_chans['consec_no_detect_frames'] = self.consec_no_detect_frames
        if chans is not None:
            self._telem_chans.update(chans)
        self._telem_mutex.release()


class HybridErrorSource(ErrorSource):
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
        blind: BlindErrorSource instance.
        optical: OpticalErrorSource instance.
        max_divergence: Max allowed angular separation between the topocentric target positions
            predicted by both targets.
        max_optical_no_signal_frames: Reverts to blind mode after this many frames with no target.
        state: String used as an enum to represent state: 'blind' or 'optical'.
    """

    class State(IntEnum):
        """Represents state of hybrid error source"""
        BLIND = 0
        OPTICAL = 1

    def __init__(
            self,
            mount: TelescopeMount,
            mount_model: MountModel,
            target: Target,
            camera: Camera,
            max_divergence: Angle = Angle(5.0 * u.deg),
            max_optical_no_signal_frames: int = 4,
            meridian_side: MeridianSide = MeridianSide.WEST,
        ):
        """Construct an instance of HybridErrorSource

        Args:
            mount: The mount under active control.
            mount_model: Required to transform between different reference frames.
            target: Represents the target to be tracked in BlindErrorSource.
            camera: Camera from which to capture imagery in OpticalErrorSource.
            max_divergence: Maximum allowed angular separation between target position estimated by
                the optical and blind error sources when in optical mode. When this is exceeded
                the state reverts to blind mode.
            max_optical_no_signal_frames: Maximum number of consecutive camera frames with no
                target detected before reverting to blind mode.
            meridian_side: Mount will stay on this side of the meridian.
        """
        self.blind = BlindErrorSource(
            mount=mount,
            mount_model=mount_model,
            target=target,
            meridian_side=meridian_side
        )
        self.optical = OpticalErrorSource(
            camera=camera,
            mount=mount,
            mount_model=mount_model,
            meridian_side=meridian_side
        )
        self.max_divergence = max_divergence
        self.max_optical_no_signal_frames = max_optical_no_signal_frames
        self.state = self.State.BLIND
        print('Hybrid error source starting in blind tracking state')
        super().__init__()


    def compute_error(self) -> PointingError:
        """Compute encoder error terms using both blind and optical methods and pick one to return.

        Pointing error computation is attempted for both optical and blind error sources on each
        invocation. The state is updated based on the presence of a target detected by the camera
        and the separation angle between the target position as predicted from ephemeris and by the
        camera. The pointing error terms corresponding to the state after all updates is returned.

        Returns:
            Error terms for each mount axis.
        """

        blind_error = self.blind.compute_error()

        try:
            self.optical.camera_timeout = inf if self.state == self.State.OPTICAL else 0.0
            optical_error = self.optical.compute_error()
        except ErrorSource.NoSignalException:
            if self.state == self.State.BLIND:
                self._set_telem_channels()
                return blind_error
            if self.optical.consec_no_detect_frames >= self.max_optical_no_signal_frames:
                print('Lost target in camera, switching to blind tracking')
                self.state = self.State.BLIND
                self._set_telem_channels()
                return blind_error
            raise

        # Get angle between optical and blind target position solutions. This is a measure of how
        # far the two solutions have diverged. A large divergence angle could mean that the
        # computer vision algorithm is not tracking the correct object.
        divergence_angle = self.optical.target_position.separation(self.blind.target_position)

        if self.state == self.State.BLIND and divergence_angle < self.max_divergence:
            print('A target is in view, switching to optical tracking')
            self.state = self.State.OPTICAL
        elif self.state == self.State.OPTICAL and divergence_angle > self.max_divergence:
            print('Solutions diverged, switching to blind tracking')
            self.state = self.State.BLIND

        telem = {}
        telem['divergence_angle'] = divergence_angle.deg
        self._set_telem_channels(telem)

        if self.state == self.State.BLIND:
            return blind_error
        else:
            return optical_error

    def _set_telem_channels(self, chans: Optional[Dict] = None) -> None:
        """Set telemetry dict polled by telemetry thread"""
        self._telem_mutex.acquire()
        self._telem_chans = {}
        self._telem_chans['state'] = self.state.value
        if chans is not None:
            self._telem_chans.update(chans)
        self._telem_mutex.release()
