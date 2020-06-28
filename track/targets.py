"""targets for use in telescope tracking control loop"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from abc import ABC, abstractmethod
from functools import lru_cache
from math import inf
import threading
import numpy as np
from astropy.coordinates import (Angle, EarthLocation, SkyCoord, AltAz, Longitude,
    UnitSphericalRepresentation)
from astropy.time import Time
from astropy import units as u
from astropy.units import Quantity
import ephem
import cv2
from track.cameras import Camera
from track.compvis import find_features, PreviewWindow
from track.model import MountModel
from track.mounts import MeridianSide, MountEncoderPositions, TelescopeMount
from track.telem import TelemSource


# TODO: Make this the return type of Target.get_position()
class TargetPosition(NamedTuple):
    """Position of a target at a specific time.

    Attributes:
        time: Time at which the target is expected to be located at this position.
        position_topo: Topographical target position (azimuth and altitude).
        position_enc: Mount encoder positions corresponding to target's apparent position.
    """
    time: Time
    topo: SkyCoord
    enc: MountEncoderPositions


class TargetVelocity(NamedTuple):
    """Instantaneous direction and angular rate of a target.

    Attributes:
        position_angle: The position angle defining the direction of motion.
        rate: The rate of motion (example units: deg/s) along a great circle.
    """
    position_angle: Angle
    rate: Quantity


class Target(ABC):
    """Abstract base class providing a common interface for targets to be tracked."""

    class IndeterminatePosition(Exception):
        """Raised when computing the target position is impossible.

        This may happen, for example, when a computer vision algorithm is unable to detect the
        target in the camera frame.
        """

    @abstractmethod
    def get_position(self, t: Time) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Get the apparent position of the target for the specified time.

        Args:
            t: The time for which the position should correspond.

        Returns:
            A tuple containing:
            1. A SkyCoord object having AltAz frame giving the apparent position of the object in
               the topocentric reference frame
            2. An instance of MountEncoderPositions giving the apparent position of the object in
               terms of the mount's encoder positions.

        Raises:
            IndeterminatePosition if the target position cannot be determined.
        """

    def process_sensor_data(self) -> None:
        """Get and process data from any sensors associated with this target type.

        This method will be called once near the beginning of the control cycle. Reading of sensor
        data and processing of that data into intermediate state should be done in this method. The
        code should be optimized such that calls to get_position() are as fast as practical. If no
        sensors are associated with this target type there is no need to override this default
        no-op implementation.
        """
        pass


class FixedTopocentricTarget(Target):
    """A target at a fixed topocentric position.

    Targets of this type remain at a fixed apparent position in the sky. An example might be a tall
    building. These objects do not appear to move as the Earth rotates and do not have any
    significant velocity relative to the observer.
    """

    def __init__(self, coord: SkyCoord, mount_model: MountModel, meridian_side: MeridianSide):
        """Construct an instance of FixedTopocentricTarget.

        Args:
            coord: An instance of SkyCoord having AltAz frame.
            mount_model: An instance of class MountModel for coordinate system conversions.
            meridian_side: Desired side of mount-relative meridian.
        """
        if not isinstance(coord.frame, AltAz):
            raise TypeError('frame of coord must be AltAz')
        self.position_topo = coord
        self.position_enc = mount_model.topocentric_to_encoders(coord, meridian_side)

    def get_position(self, t: Optional[Time] = None) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Since the topocentric position is fixed the t argument is ignored"""
        return (self.position_topo, self.position_enc)


class AcceleratingMountAxisTarget(Target):
    """A target that accelerates at a constant rate in one or both mount axes.

    This target is intented for testing the control system's ability to track an accelerating
    target with reasonably small steady-state error.
    """

    def __init__(
            self,
            mount_model: MountModel,
            initial_encoder_positions: MountEncoderPositions,
            axis_accelerations: Tuple[float, float],
        ):
        """Construct an AcceleratingMountAxisTarget.

        Initial velocity of the target is zero in both axes. Acceleration begins the moment this
        constructor is called and continues forever without limit as currently implemented.

        Args:
            mount_model: An instance of class MountModel for coordinate system conversions.
            initial_encoder_positions: The starting positions of the mount encoders. Note that if
                axis 1 starts out pointed at the pole and has acceleration 0 axis 0 may not behave
                as expected since the pole is a singularity. A work-round is to set the initial
                encoder position for axis 1 to a small offset from the pole, or to use a non-zero
                acceleration for axis 1.
            axis_accelerations: A tuple of two floats giving the accelerations in axis 0 and axis
                1, respectively, in degrees per second squared (negative okay).
        """
        self.mount_model = mount_model
        self.accel = axis_accelerations
        self.time_start = None
        self.initial_positions = initial_encoder_positions

    def get_position(self, t: Time) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Gets the position of the simulated target for a specific time."""

        if self.time_start is None:
            # Don't do this in constructor because it may be a couple seconds between when the
            # constructor is called until the first call to this method.
            self.time_start = t

        time_elapsed = (t - self.time_start).sec
        position_enc = MountEncoderPositions(
            Longitude((self.initial_positions[0].deg + self.accel[0] * time_elapsed**2) * u.deg),
            Longitude((self.initial_positions[1].deg + self.accel[1] * time_elapsed**2) * u.deg),
        )
        position_topo = self.mount_model.encoders_to_topocentric(position_enc)
        return (position_topo, position_enc)


class PyEphemTarget(Target):
    """A target using the PyEphem package"""

    def __init__(
            self,
            target,
            location: EarthLocation,
            mount_model: MountModel,
            meridian_side: MeridianSide
        ):
        """Init a PyEphem target

        This target type uses PyEphem, the legacy package for ephemeris calculations.

        Args:
            target: One of the various PyEphem body objects. Objects in this category should have
                a compute() method.
            location: Location of the observer.
            mount_model: An instance of class MountModel for coordinate system conversions.
            meridian_side: Desired side of mount-relative meridian.
        """

        self.target = target

        # Create a PyEphem Observer object for the given location
        self.observer = ephem.Observer()
        self.observer.lat = location.lat.rad
        self.observer.lon = location.lon.rad
        self.observer.elevation = location.height.to_value(u.m)

        self.mount_model = mount_model
        self.meridian_side = meridian_side


    @lru_cache(maxsize=128)  # cache results to avoid re-computing unnecessarily
    def get_position(self, t: Time) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Get apparent position of this target"""
        self.observer.date = ephem.Date(t.datetime)
        self.target.compute(self.observer)
        position_topo = SkyCoord(self.target.az * u.rad, self.target.alt * u.rad, frame='altaz')
        position_enc = self.mount_model.topocentric_to_encoders(position_topo, self.meridian_side)
        return (position_topo, position_enc)


class CameraTarget(Target, TelemSource):
    """Target based on computer vision detection of objects in a guide camera.

    This class identifies a target in a camera frame using computer vision. The target position in
    the camera frame is transformed to full-sky coordinate systems.
    """

    class TargetPositionMount(NamedTuple):
        time: Time
        position: SkyCoord  # in mount frame

    def __init__(
            self,
            camera: Camera,
            mount: TelescopeMount,
            mount_model: MountModel,
            meridian_side: Optional[MeridianSide] = None,
            camera_timeout: float = inf,
        ):
        """Construct an instance of CameraTarget

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

        if meridian_side is not None:
            self.meridian_side = meridian_side
        else:
            _, self.meridian_side = mount_model.encoder_to_spherical(mount.get_position())

        frame_height, frame_width = camera.frame_shape
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)
        self.preview_window = PreviewWindow(frame_width, frame_height)

        self.position_deque = deque(maxlen=2)
        self.camera_thread = Thread(
            target=self._camera_thread,
            args=(frame_width, frame_height),
        )
        self.camera_thread_stop = False
        self.camera_thread.start()

        self._telem_mutex = threading.Lock()
        self._telem_chans = {}

    def _camera_to_mount_position(
            self,
            target_x: Angle,
            target_y: Angle
        ) -> Tuple[SkyCoord, Dict]:
        """Transform from target position in camera frame to position in mount frame

        Args:
            target_x: Target position in camera's x-axis
            target_y: Target position in camera's y-axis

        Returns:
            Tuple with target position in mount frame and dict of telemetry channels.
        """

        # angular separation and direction from center of camera frame to target
        target_position_cam = target_x.deg + 1j*target_y.deg
        target_offset_magnitude = Angle(np.abs(target_position_cam) * u.deg)
        target_direction_cam = Angle(np.angle(target_position_cam) * u.rad)

        # Current position of mount is assumed to be the position of the center of the camera frame
        mount_enc_positions = self.mount.get_position()
        mount_coord, mount_meridian_side = self.mount_model.encoder_to_spherical(
            mount_enc_positions
        )

        # Find the position of the target relative to the mount position
        target_position_angle = self.mount_model.guide_cam_orientation - target_direction_cam
        if mount_meridian_side == MeridianSide.EAST:
            # camera orientation flips when crossing the pole
            target_position_angle += 180*u.deg
        target_coord = SkyCoord(mount_coord).directional_offset_by(
            position_angle=target_position_angle,
            separation=target_offset_magnitude
        )

        telem_chans = {}
        telem_chans['error_mag'] = target_offset_magnitude.deg
        telem_chans['target_direction_cam'] = target_direction_cam.deg
        telem_chans['target_position_angle'] = target_position_angle.deg
        for axis in self.mount.AxisName:
            telem_chans[f'mount_enc_{axis}'] = mount_enc_positions[axis].deg

        return target_coord, telem_chans

    def _get_keypoint_xy(self, keypoint: cv2.KeyPoint) -> Tuple[float, float]:
        """Get the x/y coordinates of a keypoint in the camera frame.

        Transform keypoint position to a Cartesian coordinate system defined such that (0,0) is the
        center of the camera frame, +Y points toward the top of the frame, and +X points toward the
        right edge of the frame. Keypoint indices start from zero in the upper-left corner,
        therefore the horizontal index increases in the +X direction and the vertical index
        increases in the -Y direction.

        Args:
            keypoint: A keypoint defining a position in a camera frame.

        Returns:
            A tuple with (x_position, y_position) with units of pixels.
        """
        keypoint_x_px = keypoint.pt[0] - self.frame_center_px[0]
        keypoint_y_px = self.frame_center_px[1] - keypoint.pt[1]
        return keypoint_x_px, keypoint_y_px


    def _keypoint_nearest_center_frame(self, keypoints: List[cv2.KeyPoint]) -> cv2.KeyPoint:
        """Find the keypoint closes to the center of the frame from a list of keypoints"""
        min_dist = None
        target_keypoint = None
        for keypoint in keypoints:
            keypoint_x_px, keypoint_y_px = self._get_keypoint_xy(keypoint)
            keypoint_dist_from_center_px = np.abs(keypoint_x_px + 1j*keypoint_y_px)

            if min_dist is None or keypoint_dist_from_center_px < min_dist:
                target_keypoint = keypoint
                min_dist = keypoint_dist_from_center_px

        return target_keypoint

    # TODO: Update docstring
    @lru_cache(maxsize=128)  # cache results to avoid re-computing unnecessarily
    def get_position(self, t: Time) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Compute target position using computer vision and a camera.

        The positions are computed using this rough procedure:
        1) A frame is obtained from the camera
        2) Computer vision algorithms are applied to identify bright objects
        3) The bright object nearest the center of the camera frame is assumed to be the target
        4) The error vector for the selected object is computed in the camera's frame
        5) The error vector is transformed from camera frame to mount encoder positions and altaz

        Args:
            t:

        Returns:
            Topocentric position of the target and corresponding mount encoder positions.

        Raises:
            IndeterminatePosition if a target could not be identified in the camera frame.
            ValueError if t is not None.
        """

        if self.target_velocity is None:
            raise self.IndeterminatePosition('No velocity estimate')

        if len(self.target_deque) < 2:
            raise self.IndeterminatePosition('No velocity estimate')

        position_last_frame = self.target_deque[1]
        time_diff = (t - position_last_frame.time).to(u.s)
        position_mount = position_last_frame.position.directional_offset_by(
            position_angle=self.target_velocity.position_angle,
            separation=(self.target_velocity.rate * time_diff),
        )

        position_enc = self.model.spherical_to_encoder(position_mount, self.meridian_sid)
        position_topo = self.model.spherical_to_topocentric(position_topo)

        return position_topo, position_enc

    def process_sensor_data(self) -> None:
        """Get frame from camera, detect targets, do coordinate transforms"""

        frame = self.camera.get_frame(timeout=self.camera_timeout)
        time_frame = Time.now()

        if frame is None:
            self._set_telem_channels()
            return

        keypoints = find_features(frame)

        if not keypoints:
            self.preview_window.show_annotated_frame(frame)
            self._set_telem_channels()
            return

        # assume that the target is the keypoint nearest the center of the camera frame
        target_keypoint = self._keypoint_nearest_center_frame(keypoints)

        self.preview_window.show_annotated_frame(frame, keypoints, target_keypoint)

        # convert target position units from pixels to degrees
        target_x_px, target_y_px = self._get_keypoint_xy(target_keypoint)
        target_x = Angle(target_x_px * self.camera.pixel_scale * self.camera.binning * u.deg)
        target_y = Angle(target_y_px * self.camera.pixel_scale * self.camera.binning * u.deg)

        # transform to world coordinates
        target_position_mount, telem = self._camera_to_mount_position(target_x, target_y)
        telem['target_cam_x'] = target_x.deg
        telem['target_cam_y'] = target_y.deg

        target_position = TargetPositionMount(time_frame, target_position_mount)
        self.position_deque.append(target_position)

        if len(self.position_deque) < 2:
            self.target_velocity = None
            self._set_telem_channels(telem)
            return

        # estimate target velocity based on positions from last two frames
        # TODO: There may be better ways to do this...relying on just the two most recent
        # frames may be less robust than using multiple frames, or using two frames that are
        # more separated in time.
        target_position_earlier = self.position_deque[0]
        time_elapsed = (target_position.time - target_position_earlier.time).to(u.s)
        # TODO: threshold is a magic number
        if time_elapsed > 1*u.s:
            self.target_velocity = None
            self._set_telem_channels(telem)
            return

        self.target_velocity = TargetVelocity(
            position_angle=target_position_earlier.position.position_angle(target_position.position),
            rate=(separation(target_position.topo, target_position_earlier.topo)
                / time_elapsed)
        )

        self._set_telem_channels(telem)

    def _set_telem_channels(self, chans: Optional[Dict] = None) -> None:
        """Set telemetry dict polled by telemetry thread"""
        self._telem_mutex.acquire()
        self._telem_chans = {}
        if chans is not None:
            self._telem_chans.update(chans)
        self._telem_mutex.release()

    def get_telem_channels(self):
        """Called by telemetry polling thread -- see TelemSource abstract base class"""
        # Protect dict copy with mutex since this method is called from another thread
        self._telem_mutex.acquire()
        chans = self._telem_chans.copy()
        self._telem_mutex.release()
        return chans
