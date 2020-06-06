"""targets for use in telescope tracking control loop"""

from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from functools import lru_cache
from math import inf
import threading
import numpy as np
from astropy.coordinates import (Angle, EarthLocation, SkyCoord, AltAz, Longitude,
    UnitSphericalRepresentation)
from astropy.time import Time
from astropy import units as u
import ephem
import cv2
from track.cameras import Camera
from track.compvis import find_features, PreviewWindow
from track.model import MountModel
from track.mounts import MeridianSide, MountEncoderPositions, TelescopeMount
from track.telem import TelemSource


class Target(ABC):
    """Abstract base class providing a common interface for targets to be tracked."""

    class IndeterminatePosition(Exception):
        """Raised when computing the target position is impossible.

        This may happen, for example, when a computer vision algorithm is unable to detect the
        target in the camera frame.
        """

    @property
    @abstractmethod
    def supports_prediction(self) -> bool:
        """True if the class supports `get_position()` for specific times.

        Certain instances of Target may only be able to determine the target position at the
        moment `get_position()` is called. When this is the case, this property will be False and
        `get_position()` will only succeed when the argument is left unspecified.
        """

    @abstractmethod
    def get_position(self, t: Optional[Time] = None) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Get the apparent position of the target for the specified time.

        Args:
            t: The time for which the position should correspond. If None, the position returned
                will correspond to the time this method is called and should be nearly the same
                as calling with this argument set to Time.now().

        Returns:
            A tuple containing:
            1. A SkyCoord object having AltAz frame giving the apparent position of the object in
               the topocentric reference frame
            2. An instance of MountEncoderPositions giving the apparent position of the object in
               terms of the mount's encoder positions.

        Raises:
            IndeterminatePosition if the target position cannot be determined.
            ValueError if t is not None and the `supports_prediction` attribute is False.
        """


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

    @property
    def supports_prediction(self) -> bool:
        """Prediction is trivial since the target apparent position is fixed"""
        return True

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

    @property
    def supports_prediction(self) -> bool:
        return True

    def get_position(self, t: Optional[Time] = None) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Gets the position of the simulated target for a specific time."""

        if t is None:
            t = Time.now()

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

    @property
    def supports_prediction(self) -> bool:
        """PyEphem is able to calculate target positions for any time"""
        return True


    def get_position(self, t: Optional[Time] = None) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Get apparent position of this target"""
        return self._get_position(t if t is not None else Time.now())


    @lru_cache(maxsize=128)  # cache results to avoid re-computing unnecessarily
    def _get_position(self, t: Time) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Implementation of get_position()

        A wrapper is necessary to allow memoization caching to work properly. Without this
        wrapper, the lru_cache decorator applied to the get_position() method would cause it to
        return the same value every time it is called with the argument set to None, which is not
        the desired result.
        """
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

        if meridian_side is not None:
            self.meridian_side = meridian_side
        else:
            _, self.meridian_side = mount_model.encoder_to_spherical(mount.get_position())

        frame_height, frame_width = camera.frame_shape
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)
        self.preview_window = PreviewWindow(frame_width, frame_height)

        self._telem_mutex = threading.Lock()
        self._telem_chans = {}

    @property
    def supports_prediction(self) -> bool:
        """Can only calculate target position using the most recent camera frame"""
        return False

    def _camera_to_mount_position(
            self,
            target_x: Angle,
            target_y: Angle
        ) -> Tuple[SkyCoord, MountEncoderPositions, Dict]:
        """Transform from target position in camera frame to positions in 3D coordinate systems

        Args:
            target_x: Target position in camera's x-axis
            target_y: Target position in camera's y-axis

        Returns:
            Target position in topocentric frame, corresponding mount encoder positions, and dict
            of telemetry channels.
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
        ).represent_as(UnitSphericalRepresentation)

        # convert target position back to mount encoder positions
        target_position_enc = self.mount_model.spherical_to_encoder(
            coord=target_coord,
            meridian_side=self.meridian_side,
        )

        target_position_topo = self.mount_model.spherical_to_topocentric(target_coord)

        telem_chans = {}
        telem_chans['error_mag'] = target_offset_magnitude.deg
        telem_chans['target_direction_cam'] = target_direction_cam.deg
        telem_chans['target_position_angle'] = target_position_angle.deg
        for axis in self.mount.AxisName:
            telem_chans[f'target_enc_{axis}'] = target_position_enc[axis].deg
            telem_chans[f'mount_enc_{axis}'] = mount_enc_positions[axis].deg

        return target_position_topo, target_position_enc, telem_chans

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


    def get_position(self, t: None = None) -> Tuple[SkyCoord, MountEncoderPositions]:
        """Compute pointing error using computer vision and a camera.

        The encoder error terms are computed using this rough procedure:
        1) A frame is obtained from the camera
        2) Computer vision algorithms are applied to identify bright objects
        3) The bright object nearest the center of the camera frame is assumed to be the target
        4) The error vector for the selected object is computed in the camera's frame
        5) The error vector is transformed from camera frame to mount encoder positions and altaz

        Args:
            t: Must be None. The position returned is always for the most recently received camera
                frame, which corresponds roughly to the time this method is called.

        Returns:
            Topocentric position of the target and corresponding mount encoder positions.

        Raises:
            IndeterminatePosition if a target could not be identified in the camera frame.
            ValueError if t is not None.
        """
        if t is not None:
            raise ValueError('prediction not supported')

        frame = self.camera.get_frame(timeout=self.camera_timeout)

        if frame is None:
            self._set_telem_channels()
            raise self.IndeterminatePosition('No frame was available')

        keypoints = find_features(frame)

        if not keypoints:
            self.preview_window.show_annotated_frame(frame)
            self._set_telem_channels()
            raise self.IndeterminatePosition('No target identified')

        # assume that the target is the keypoint nearest the center of the camera frame
        target_keypoint = self._keypoint_nearest_center_frame(keypoints)

        self.preview_window.show_annotated_frame(frame, keypoints, target_keypoint)

        # convert target position units from pixels to degrees
        target_x_px, target_y_px = self._get_keypoint_xy(target_keypoint)
        target_x = Angle(target_x_px * self.camera.pixel_scale * self.camera.binning * u.deg)
        target_y = Angle(target_y_px * self.camera.pixel_scale * self.camera.binning * u.deg)

        # transform to mount encoder error terms
        position_topo, position_enc, telem = self._camera_to_mount_position(target_x, target_y)

        telem['target_cam_x'] = target_x.deg
        telem['target_cam_y'] = target_y.deg

        self._set_telem_channels(telem)

        return position_topo, position_enc

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
