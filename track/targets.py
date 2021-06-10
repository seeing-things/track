"""targets for use in telescope tracking control loop"""

from math import inf
from typing import List, Optional, Tuple, NamedTuple
from abc import abstractmethod
from functools import lru_cache
from datetime import datetime
import dateutil
import numpy as np
from configargparse import Namespace
from astropy.coordinates import Angle, EarthLocation, SkyCoord, AltAz, Longitude
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy.time import Time
from astropy import units as u
import ephem
import cv2
from influxdb_client import Point
from track import cameras
from track.config import ArgParser
from track.cameras import Camera
from track.compvis import find_features, PreviewWindow
from track.model import MountModel
from track.mounts import MeridianSide, MountEncoderPositions, TelescopeMount
from track.telem import TelemLogger


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


class Target:
    """Abstract base class providing a common interface for targets to be tracked."""

    class IndeterminatePosition(Exception):
        """Raised when computing the target position is impossible.

        This may happen, for example, when a computer vision algorithm is unable to detect the
        target in the camera frame.
        """

    @abstractmethod
    def get_position(self, t: Time) -> TargetPosition:
        """Get the apparent position of the target for the specified time.

        Args:
            t: The time for which the position should correspond, if possible. For some targets
                the position can be found at this exact time. For others it may not be possible to
                predict the position in the past or in the future. The `time` field of the return
                tuple will be populated to indicate the actual time that corresponds to the return
                value's position.

        Returns:
            The target position as an instance of TargetPosition.

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


class FixedMountEncodersTarget(Target):
    """A target at fixed mount encoder positions.

    Targets of this type remain at fixed mount encoder positions. This is similar to the
    `FixedTopocentricTarget` class except that it avoids numerical precision issues for positions
    near the mount pole, if the particular encoder position of the first axis (typically the right
    ascension or azimuth axis) is important.
    """

    def __init__(self, enc: MountEncoderPositions, mount_model: MountModel):
        """Construct an instance of FixedMountEncodersTarget.

        Args:
            enc: A set of mount encoder positions.
            mount_model: An instance of class MountModel for coordinate system conversions.
        """
        self.position_enc = enc
        self.position_topo = mount_model.encoders_to_topocentric(enc)

    def get_position(self, t: Optional[Time] = None) -> TargetPosition:
        """Since the position is fixed the t argument is ignored"""
        return TargetPosition(t, self.position_topo, self.position_enc)


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

    def get_position(self, t: Optional[Time] = None) -> TargetPosition:
        """Since the topocentric position is fixed the t argument is ignored"""
        return TargetPosition(t, self.position_topo, self.position_enc)


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

    def get_position(self, t: Time) -> TargetPosition:
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
        return TargetPosition(t, position_topo, position_enc)


class OverheadPassTarget(Target):
    """A target that passes directly overhead at a steady 1 degree per second horizon-to-horizon.
    """

    def __init__(
            self,
            mount_model: MountModel,
            meridian_side: MeridianSide,
        ):
        """Construct an OverheadPassTarget.

        Args:
            mount_model: An instance of class MountModel for coordinate system conversions.
            meridian_side: Desired side of mount-relative meridian.
        """
        self.mount_model = mount_model
        self.meridian_side = meridian_side
        self.time_start = Time.now()
        self.position_start = SkyCoord(90*u.deg, -20*u.deg, frame='altaz')
        self.position_angle = self.position_start.position_angle(
            SkyCoord(0*u.deg, 90*u.deg, frame='altaz')
        )

    @lru_cache(maxsize=128)  # cache results to avoid re-computing unnecessarily
    def get_position(self, t: Time) -> TargetPosition:
        """Gets the position of the simulated target for a specific time."""
        time_elapsed = (t - self.time_start).sec
        separation = time_elapsed*u.deg  # 1 deg/s

        position_topo = self.position_start.directional_offset_by(self.position_angle, separation)
        position_enc = self.mount_model.topocentric_to_encoders(position_topo, self.meridian_side)

        return TargetPosition(t, position_topo, position_enc)


class FlightclubLaunchTrajectoryTarget(Target):
    """A target that follows a trajectory predicted by a flightclub.io simulation"""

    def __init__(
            self,
            filename: str,
            time_t0: Time,
            mount_model: MountModel,
            meridian_side: MeridianSide,
        ):
        self.time_t0 = time_t0
        self.mount_model = mount_model
        self.meridian_side = meridian_side

        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        self.times_from_t0 = data[:, 0]
        self.alt = data[:, 1]
        self.az = np.degrees(np.unwrap(np.radians(data[:, 2])))

    @lru_cache(maxsize=128)  # cache results to avoid re-computing unnecessarily
    def get_position(self, t: Time) -> TargetPosition:
        """Get apparent position of this target"""

        # use linear interpolation between available trajectory data points
        time_from_t0 = (t - self.time_t0).to_value('sec')
        position_az = np.interp(time_from_t0, self.times_from_t0, self.az)
        position_alt = np.interp(time_from_t0, self.times_from_t0, self.alt)

        position_topo = SkyCoord(position_az * u.deg, position_alt * u.deg, frame='altaz')
        position_enc = self.mount_model.topocentric_to_encoders(position_topo, self.meridian_side)
        return TargetPosition(t, position_topo, position_enc)


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
    def get_position(self, t: Time) -> TargetPosition:
        """Get apparent position of this target"""
        self.observer.date = ephem.Date(t.datetime)
        self.target.compute(self.observer)
        position_topo = SkyCoord(self.target.az * u.rad, self.target.alt * u.rad, frame='altaz')
        position_enc = self.mount_model.topocentric_to_encoders(position_topo, self.meridian_side)
        return TargetPosition(t, position_topo, position_enc)


class CameraTarget(Target):
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
            telem_logger: Optional[TelemLogger] = None,
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
            telem_logger: Optional instance of `TelemLogger` to which telemetry points will be
                written.
        """
        self.camera = camera
        self.camera_timeout = camera_timeout
        self.mount = mount
        self.mount_model = mount_model
        self.telem_logger = telem_logger
        self.guide_cam_align_error = mount_model.model_param_set.guide_cam_align_error
        self.guide_cam_align_error_px = (
            self.guide_cam_align_error.deg
            / (self.camera.pixel_scale * self.camera.binning)
        )

        if meridian_side is not None:
            self.meridian_side = meridian_side
        else:
            _, self.meridian_side = mount_model.encoder_to_spherical(mount.get_position())

        frame_height, frame_width = camera.frame_shape
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)
        self.preview_window = PreviewWindow(frame_width, frame_height)

        self.target_position = None

    def camera_to_directional_offset(
            self,
            target_x: Angle,
            target_y: Angle,
            mount_meridian_side: MeridianSide,
        ) -> Tuple[Angle, Angle]:
        """Transform from a position in camera frame to a magnitude and position angle

        Args:
            target_x: Target position in camera's x-axis
            target_y: Target position in camera's y-axis
            mount_meridian_side: The mount's meridian side at the time when the camera frame was
                captured

        Returns:
            A tuple containing the target position offset magnitude and angle in the mount-relative
            frame, as would be passed to `SkyCoord.directional_offset_by()`.
        """

        # position of target relative to center of camera frame
        target_position_cam = target_x + 1j*target_y

        # angular separation and direction from approximate center of main OTA camera to target
        target_position_main_ota = target_position_cam - self.guide_cam_align_error
        target_offset_magnitude = np.abs(target_position_main_ota)
        target_direction_cam = np.angle(target_position_main_ota)

        # Find the position angle from the mount's position to the target's position
        target_position_angle = self.mount_model.guide_cam_orientation - target_direction_cam
        if mount_meridian_side == MeridianSide.EAST:
            # camera orientation flips when crossing the pole
            target_position_angle += 180*u.deg

        return target_offset_magnitude, target_position_angle

    def _camera_to_mount_position(
            self,
            target_x: Angle,
            target_y: Angle,
        ) -> UnitSphericalRepresentation:
        """Transform from target position in camera frame to position in mount frame

        Args:
            target_x: Target position in camera's x-axis
            target_y: Target position in camera's y-axis

        Returns:
            Target position in mount frame
        """

        # Current position of mount is assumed to be the position of the center of the camera frame
        # FIXME: This isn't going to be very accurate since we are not getting the mount position
        # as close in time as possible to when the camera frame was captured. What we *really* need
        # are a couple of things:
        # 1. A reasonable estimate of the actual time when the camera frame was captured (not just
        #    when the function to get the frame data returns, since there is hidden latency there)
        # 2. A way to get the mount's position corresponding to that past time. We have the ability
        #    to predict future mount positions but not a way to go back and get past positions
        #    (and interpolate between them). Maybe it would be reasonable for the mount to cache
        #    past positions in a limited-length deque with timestamps to make this past position
        #    lookup possible. Maybe add another argument to `get_position()` to specify the desired
        #    past time.
        mount_enc_positions = self.mount.get_position()
        mount_coord, mount_meridian_side = self.mount_model.encoder_to_spherical(
            mount_enc_positions
        )

        target_offset_magnitude, target_position_angle = self.camera_to_directional_offset(
            target_x,
            target_y,
            mount_meridian_side,
        )

        target_coord = SkyCoord(mount_coord).directional_offset_by(
            position_angle=target_position_angle,
            separation=target_offset_magnitude
        ).represent_as(UnitSphericalRepresentation)

        return target_coord

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
            A tuple containing the (X, Y) position of the target with units of pixels where the
            origin is the center of the camera frame.
        """
        keypoint_x_px = keypoint.pt[0] - self.frame_center_px[0]
        keypoint_y_px = self.frame_center_px[1] - keypoint.pt[1]
        return keypoint_x_px, keypoint_y_px


    def _keypoint_nearest_desired_target_position(
            self,
            keypoints: List[cv2.KeyPoint]
        ) -> cv2.KeyPoint:
        """Find the keypoint closest to the desired target position from a list of keypoints

        Args:
            keypoints: List of keypoints to filter.

        Returns:
            A single keypoint that is nearest to the position in the guide camera frame
            corresponding to the assumed center of the main OTA camera frame.
        """
        min_dist = None
        target_keypoint = None
        for keypoint in keypoints:
            keypoint_x_px, keypoint_y_px = self._get_keypoint_xy(keypoint)
            keypoint_px = keypoint_x_px + 1j*keypoint_y_px
            keypoint_dist_from_desired_px = np.abs(keypoint_px - self.guide_cam_align_error_px)

            if min_dist is None or keypoint_dist_from_desired_px < min_dist:
                target_keypoint = keypoint
                min_dist = keypoint_dist_from_desired_px

        return target_keypoint

    def get_position(self, t: Time) -> TargetPosition:
        """Compute target position using computer vision and a camera.

        Args:
            t: Ignored. Position returned always corresponds to the most recent time
                process_sensor_data() was called and successfully identified a target.

        Returns:
            Topocentric position of the target and corresponding mount encoder positions.

        Raises:
            IndeterminatePosition if a target could not be identified in the most recent camera
                frame.
        """
        if self.target_position is None:
            raise self.IndeterminatePosition('No target detected in most recent frame')
        return self.target_position

    def process_camera_frame(self) -> Tuple[Time, Angle, Angle]:
        """Get frame from camera and find target using computer vision

        Args:
            telem: Dict into which telemetry channels will be added

        Returns:
            Tuple containing:
            - The approximate time that the camera frame was captured.
            - The position of the target within the camera frame where the first element is the X
              position and the second is the Y position and the origin is the center of the camera
              frame.
        """
        # This time isn't going to be exceptionally accurate, but unfortunately most cameras do not
        # provide a means of determining the exact time when the frame was captured by the sensor.
        # There are probably ways to estimate the frame time more accurately but this is likely
        # good enough.
        target_time = Time.now()
        frame = self.camera.get_frame(timeout=self.camera_timeout)

        if frame is None:
            raise self.IndeterminatePosition('Timeout waiting for frame from camera')

        keypoints = find_features(frame)

        if not keypoints:
            self.preview_window.show_annotated_frame(frame)
            raise self.IndeterminatePosition('No target detected in most recent frame')

        # assume that the target is the keypoint nearest the center of the camera frame
        target_keypoint = self._keypoint_nearest_desired_target_position(keypoints)

        self.preview_window.show_annotated_frame(frame, keypoints, target_keypoint)

        # convert target position units from pixels to degrees
        target_x_px, target_y_px = self._get_keypoint_xy(target_keypoint)
        target_x = Angle(target_x_px * self.camera.pixel_scale * self.camera.binning * u.deg)
        target_y = Angle(target_y_px * self.camera.pixel_scale * self.camera.binning * u.deg)

        if self.telem_logger is not None:
            p = Point('camera_target')
            p.field('x', target_x.deg)
            p.field('y', target_y.deg)
            p.tag('units', 'degrees')
            p.tag('class', type(self).__name__)
            p.time(target_time.to_datetime())
            self.telem_logger.post_points(p)

        return target_time, target_x, target_y

    def process_sensor_data(self) -> None:
        """Process a new camera frame and cache the computed target position in this object.

        The positions are computed using this rough procedure:
        1) A frame is obtained from the camera
        2) Computer vision algorithms are applied to identify bright objects
        3) The bright object nearest the center of the camera frame is assumed to be the target
        4) The centroid position of the target blob is transformed from camera frame to topocentric
           frame and to mount encoder positions. These are cached in this object.
        """
        try:
            target_time, target_x, target_y = self.process_camera_frame()
        except self.IndeterminatePosition:
            self.target_position = None
            return

        # transform to world coordinates
        position_mount = self._camera_to_mount_position(target_x, target_y)

        position_enc = self.mount_model.spherical_to_encoder(
            position_mount,
            self.meridian_side
        )
        position_topo = self.mount_model.spherical_to_topocentric(position_mount)
        self.target_position = TargetPosition(target_time, position_topo, position_enc)


class SensorFusionTarget(Target):
    """Uses sensor fusion to combine data from a `CameraTarget` and another `Target`"""

    def __init__(
            self,
            blind_target: Target,
            camera_target: CameraTarget,
            mount: TelescopeMount,
            model: MountModel,
            meridian_side: MeridianSide,
            filter_gain: float = 5e-2,
            bias_mag_limit: Angle = Angle(1.0*u.deg),
            telem_logger: Optional[TelemLogger] = None,
        ):
        """Construct an instance of SensorFusionTarget

        Args:
            blind_target: A `Target` that is not `CameraTarget`. This should predict the position
                of the target with enough accuracy that the target is visible in the camera.
            camera_target: An instance of `CameraTarget`. When the target is detected by the camera
                the bias in the blind target will be estimated and removed.
            mount: Provides an interface to the mount.
            model: Mount alignment model.
            meridian_side: Side of mount meridian to use.
            filter_gain: The gain of the integrator filter applied to the target offset detected by
                camera. The output of the integrator is the estimate of the blind target bias.
            bias_mag_limit: The magnitude of the blind target bias estimate is limited to this
                value to prevent it from growing excessively large.
        """

        self.blind_target = blind_target
        self.camera_target = camera_target
        self.mount = mount
        self.model = model
        self.meridian_side = meridian_side
        self.blind_target_bias = 0.0
        self.filter_gain = filter_gain
        self.bias_mag_limit = bias_mag_limit
        self.telem_logger = telem_logger

    # Cache results to avoid re-computing unnecessarily. Strictly the cache should be cleared each
    # time the `blind_target_bias` member variable is updated but this is intentionally ignored to
    # reduce computational load. The ill effects of this seem to be minimal.
    @lru_cache(maxsize=128)
    def get_position(self, t: Time) -> TargetPosition:
        """Get the apparent position of the target for the specified time.

        Args:
            t: The time for which the position should correspond.

        Returns:
            The target position as an instance of TargetPosition.

        Raises:
            IndeterminatePosition if the target position cannot be determined.
        """

        # Only the topocentric position of the target is needed here since the mount encoder
        # positions will be re-computed later in this method. A potential optimization is to find
        # a way to prevent the blind target from generating this unused output.
        position_target_blind = self.blind_target.get_position(t)

        # transform blind topo position to mount frame
        position_target_blind_sph = self.model.topocentric_to_spherical(position_target_blind.topo)

        # apply directional offset using estimated bias terms
        position_target_fused_sph = SkyCoord(position_target_blind_sph).directional_offset_by(
            position_angle=np.angle(self.blind_target_bias)*u.rad,
            separation=np.abs(self.blind_target_bias)*u.deg
        ).represent_as(UnitSphericalRepresentation)

        # transform the offset mount coordinate to mount encoder and topo frames
        return TargetPosition(
            t,
            self.model.spherical_to_topocentric(position_target_fused_sph),
            self.model.spherical_to_encoder(
                coord=position_target_fused_sph,
                meridian_side=self.meridian_side
            )
        )

    def process_sensor_data(self) -> None:
        """Get camera frame and update blind target bias terms.

        This method gets a new camera frame, estimates the position of the target in the frame,
        transforms this into a complex value that represents the residual error in pointing, and
        processes this error to generate a bias term that is used to adjust the target predicted
        position used for pointing.
        """
        try:
            _, target_x, target_y = self.camera_target.process_camera_frame()
        except Target.IndeterminatePosition:
            if self.telem_logger is not None:
                p = Point('sensor_fusion')
                p.field('blind_target_bias_mag', np.abs(self.blind_target_bias))
                p.field('blind_target_bias_angle', np.degrees(np.angle(self.blind_target_bias)))
                p.tag('units', 'degrees')
                p.tag('class', type(self).__name__)
                p.time(datetime.utcnow())
                self.telem_logger.post_points(p)
            return

        # get meridian side of the mount
        mount_position = self.mount.get_position(max_cache_age=0.2)
        mount_meridian_side = self.model.encoder_to_meridian_side(mount_position)

        target_offset_mag, target_position_angle = self.camera_target.camera_to_directional_offset(
            target_x,
            target_y,
            mount_meridian_side,
        )

        target_offset = target_offset_mag.deg * np.exp(1j*target_position_angle.rad)

        # update bias term integrator
        self.blind_target_bias += self.filter_gain * target_offset

        # saturate bias magnitude so that it doesn't go off the rails
        if np.abs(self.blind_target_bias) > self.bias_mag_limit.deg:
            self.blind_target_bias = self.bias_mag_limit.deg * np.exp(
                1j*np.angle(self.blind_target_bias)
            )

        # telem['target_offset_mag'] = target_offset_mag.deg
        # telem['target_offset_position_angle'] = target_position_angle.deg
        # telem['blind_target_bias_mag'] = np.abs(self.blind_target_bias)
        # telem['blind_target_bias_angle'] = np.degrees(np.angle(self.blind_target_bias))
        # self._set_telem_channels(telem)


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments relevant to targets.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    target_group = parser.add_argument_group(
        title='General Target Options',
        description='Options that apply to all targets',
    )

    target_group.add_argument(
        '--fuse',
        help='use sensor fusion of selected target with camera',
        action='store_true',
    )
    target_group.add_argument(
        '--fusion-gain',
        help='gain for sensor fusion',
        type=float,
        default=5e-2,
    )

    subparsers = parser.add_subparsers(title='target types', dest='target_type')
    subparsers.required = True

    parser_flightclub = subparsers.add_parser(
        'flightclub',
        help='Flightclub.io trajectory CSV file'
    )
    parser_flightclub.add_argument('file', help='filename of CSV file')
    parser_flightclub.add_argument(
        'time_t0',
        help='Launch T0 in UTC. Many natural language date formats are supported.',
        type=str,
    )

    parser_tle = subparsers.add_parser('tle', help='TLE file')
    parser_tle.add_argument('file', help='filename of two-line element (TLE) target ephemeris')

    subparsers.add_parser('camera', help='follows bright target detected in camera')
    cameras.add_program_arguments(parser, profile='track')

    parser_coord_eq = subparsers.add_parser('coord-eq', help='fixed equatorial coordinate')
    parser_coord_eq.add_argument('ra', help='right ascension [deg]', type=float)
    parser_coord_eq.add_argument('dec', help='declination [deg]', type=float)

    parser_coord_topo = subparsers.add_parser('coord-topo', help='fixed topocentric coordinate')
    parser_coord_topo.add_argument('az', help='azimuth [deg]', type=float)
    parser_coord_topo.add_argument('alt', help='altitude [deg]', type=float)

    parser_star = subparsers.add_parser('star', help='named star')
    parser_star.add_argument('name', help='name of star')

    parser_solarsystem = subparsers.add_parser('solarsystem', help='named solar system body')
    parser_solarsystem.add_argument('name', help='name of planet or moon')

    subparsers.add_parser('overhead-pass', help='simulated overhead pass')


def make_target_from_args(
        args: Namespace,
        mount: TelescopeMount,
        mount_model: MountModel,
        meridian_side: MeridianSide,
        telem_logger: Optional[TelemLogger] = None,
    ) -> Target:
    """Construct the appropriate target based on the program arguments provided.

    Args:
        args: Set of program arguments.
        mount_model: Instance of MountModel.
        meridian_side: Desired side of mount-relative meridian.
    """
    if args.target_type == 'camera' and args.fuse:
        raise ValueError('Cannot fuse camera target with itself')

    if args.target_type == 'flightclub':
        print(f'In Flight Club trajectory mode using {args.file}')
        time_t0 = dateutil.parser.parse(args.time_t0)
        time_t0.replace(tzinfo=dateutil.tz.tzutc())
        print(f'T0 interpreted as {time_t0}Z')
        target = FlightclubLaunchTrajectoryTarget(
            filename=args.file,
            time_t0=Time(time_t0),
            mount_model=mount_model,
            meridian_side=meridian_side,
        )

    # Create a PyEphem Body object corresonding to the TLE file
    elif args.target_type == 'tle':
        print('In TLE file mode: \'{}\'.'.format(args.file))
        tle = []
        with open(args.file) as tlefile:
            for line in tlefile:
                tle.append(line)
        target = PyEphemTarget(
            target=ephem.readtle(tle[0], tle[1], tle[2]),
            location=mount_model.location,
            mount_model=mount_model,
            meridian_side=meridian_side,
        )

    elif args.target_type == 'camera':
        print('In camera mode')
        camera = cameras.make_camera_from_args(args, profile='track')
        target = CameraTarget(
            camera=camera,
            mount=mount,
            mount_model=mount_model,
            telem_logger=telem_logger,
        )

    # Create a PyEphem Body object corresonding to the given fixed coordinates
    elif args.target_type == 'coord-eq':
        print('In fixed equatorial coordinate mode: (RA {}, dec {}).'.format(args.ra, args.dec))
        target = PyEphemTarget(
            target=ephem.FixedBody(_ra=np.radians(args.ra), _dec=np.radians(args.dec)),
            location=mount_model.location,
            mount_model=mount_model,
            meridian_side=meridian_side,
        )

    elif args.target_type == 'coord-topo':
        print('In fixed topocentric coordinate mode: (AZ {}, ALT {}).'.format(args.az, args.alt))
        target = FixedTopocentricTarget(
            SkyCoord(args.az * u.deg, args.alt * u.deg, frame='altaz'),
            mount_model,
            meridian_side
        )

    # Get the PyEphem Body object corresonding to the given named star
    elif args.target_type == 'star':
        print('In named star mode: \'{}\''.format(args.name))
        target = PyEphemTarget(
            target=ephem.star(args.name),
            location=mount_model.location,
            mount_model=mount_model,
            meridian_side=meridian_side,
        )

    # Get the PyEphem Body object corresonding to the given named solar system body
    elif args.target_type == 'solarsystem':
        print('In named solar system body mode: \'{}\''.format(args.name))
        # pylint: disable=protected-access
        ss_objs = [name for _, _, name in ephem._libastro.builtin_planets()]
        if args.name in ss_objs:
            body_type = getattr(ephem, args.name)
            target = PyEphemTarget(
                target=body_type(),
                location=mount_model.location,
                mount_model=mount_model,
                meridian_side=meridian_side,
            )
        else:
            raise Exception(
                'The solar system body \'{}\' isn\'t present in PyEphem.'.format(args.name)
            )

    elif args.target_type == 'overhead-pass':
        target = OverheadPassTarget(mount_model, meridian_side)

    else:
        raise ValueError(f'Invalid target-type {args.target_type}')

    if args.fuse:
        print('Sensor fusion with camera enabled')
        blind_target = target
        camera = cameras.make_camera_from_args(args, profile='track')
        camera_target = CameraTarget(
            camera=camera,
            mount=mount,
            mount_model=mount_model,
            telem_logger=telem_logger,
        )
        target = SensorFusionTarget(
            blind_target=blind_target,
            camera_target=camera_target,
            mount=mount,
            model=mount_model,
            meridian_side=meridian_side,
            telem_logger=telem_logger,
            filter_gain=args.fusion_gain,
        )

    return target
