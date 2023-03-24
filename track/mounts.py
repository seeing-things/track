"""Mounts for use in telescope tracking control loop.

A set of classes that inherit from the abstract base class TelescopeMount, providing a common
API for interacting with telescope mounts. The abstraction is not perfect and some mounts may
require special interfaces that go beyond what is defined by TelescopeMount.

This abstraction was chosen over existing options such as INDI or ASCOM because the developers
found that these frameworks did not provide access to some of the low-level features of the
hardware that are required for satellite tracking applications. In particular, these frameworks
do not provide the ability to directly set the slew rates of the servos. The effort required to
modify them to suit the needs of this project was deemed to be higher than the effort required to
just roll our own solution that provides exactly what we need in a lightweight and tailored manner.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from enum import IntEnum
import logging
from typing import NamedTuple, Tuple
import time
import numpy as np
from astropy import units as u
from astropy.coordinates import Longitude
from configargparse import Namespace
import point
from track.config import ArgParser


logger = logging.getLogger(__name__)


class MeridianSide(IntEnum):
    """Indicates side of mount meridian. This is significant for equatorial mounts."""
    EAST = enum.auto()
    WEST = enum.auto()


class MountEncoderPositions(NamedTuple):
    """Set of mount physical encoder positions.

    This contains positions for telescope mounts having two motorized axes, which should cover the
    vast majority of amateur mounts. This code is intended for use with equatorial mounts,
    altitude-azimuth mounts, and equatorial mounts where the polar axis is intentionally not
    aligned with the celesital pole. Therefore the members of this tuple do not use the
    conventional axis names such as "ra" for "right-ascension" or "dec" for "declination" since
    these names are not appropriate in all scenarios.

    Both attirubutes are instances of the Astropy Longitude class since the range of values allowed
    by this class is [0, 360) degrees which corresponds nicely with the range of raw encoder values
    of most mounts after converting to degrees.

    Attributes:
        encoder_0: The axis closer to the base of the mount. For an equatorial mount this is
            usually the right ascension axis. For az-alt mounts this is usually the azimuth axis.
        encoder_1: The axis further from the base of the mount but closer to the optical tube. For
            an equatorial mount this is usually the declination axis. For az-alt mounts this is
            usually the altitude axis.
    """
    encoder_0: Longitude
    encoder_1: Longitude


class TelescopeMount(ABC):
    """Abstract base class for telescope mounts.

    This class provides some abstract methods to provide a common interface for telescope mounts.
    """

    def __enter__(self) -> TelescopeMount:
        """Support usage of this class in `with` statements."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Stop mount motion and disconnect."""
        self.shutdown()
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            logger.info(f'Handling {type(exc_value).__name__}')
            return True  # prevent exception propagation
        return False

    @property
    @abstractmethod
    def slew_accel(self) -> float:
        """Expected acceleration of when changing slew rate in degrees per second squared"""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, max_cache_age: float = 0.0) -> MountEncoderPositions:
        """Gets the current position of the mount.

        Args:
            max_cache_age: If the position has been read from the mount less than this many seconds
                ago the function may return a cached position value in lieu of reading the position
                from the mount. In cases where reading from the mount is relatively slow this may
                allow the function to return much more quickly. The default value is set to 0
                seconds, in which case the function will never return a cached value.

        Returns:
            An instance of MountEncoderPositions where the attributes are set to the current values
            of the encoders. The mapping of encoder values to attribute indices in this object
            should match the mapping used in the slew() method.
        """


    @abstractmethod
    def slew(self, axis: int, rate: float) -> None:
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis. A mount may enforce limits
        on the slew rate such as a maximum rate limit, acceleration limit, or max change in
        rate since the last command. Enforcement of such limits may result in the mount moving at
        a rate that is different than requested. The return values indicate whether a limit was
        exceeded and what slew rate was actually achieved.

        Args:
            axis: The index of the axis to which this operation applies. For typical mounts having
                two axes the allowed values would be 0 and 1. The axes should be numbered such that
                the axis closes to the pedestal or tripod is axis 0, the next-closest axis is 1,
                and so on. This mapping should match what is used in get_position.
            rate: The requested slew rate in degrees per second. The sign of the value indicates
                the direction of the slew.

        Raises:
            ValueError if the axis is out of range for the specific mount.
            ValueError if the magntiude of `rate` exceeds the max supported slew rate.
        """


    @abstractmethod
    def get_slew_rate(self, axis: int) -> float:
        """Get current slew rate on one axis.

        Gets the current slew rate on one mount axis. This should ideally return the actual slew
        rate as opposed to the last requested slew rate passed to a call to `slew()` since
        requested rates are not always achieved instantly, if at all, due to various physical
        limitations such as acceleration time and slew rate limits.

        Args:
            axis: The index of the axis to which this operation applies. Same conventions as used
                by `slew()`.

        Returns:
            The current slew rate of the axis in degrees per second.
        """


    @abstractmethod
    def safe(self) -> None:
        """Bring mount into a safe state.

        This method will do whatever is necessary to bring the mount into a safe state, such that
        there is no risk to hardware if the program terminates immediately afterward and
        communication with the mount stops. At minimum this method will stop all motion.
        """


    @abstractmethod
    def shutdown(self) -> None:
        """Bring mount into a safe state and shut down.

        This method is expected to perform the same actions as `safe()` but also perform any
        additional actions needed to close connections and free resources. After this method is
        called any calls to other methods in this class may fail.
        """


    @abstractmethod
    def no_cross_encoder_positions(self) -> MountEncoderPositions:
        """Indicate encoder positions that should not be crossed.

        For most mounts cord wrap or other physical limitations make it infeasible to slew all the
        way around in either axis. For some mounts there exists a set of positions on each axis
        that should never be crossed. For example, on a German Equatorial mount the declination
        axis should never cross the position where the optical tube is 180 degrees away from the
        celesitial pole (since this is always at or below the horizon) and the right ascension axis
        should never need to be positioned such that the counter weight is facing directly up since
        for most mounts the optical tube would collide with the tripod or pier before reaching this
        position.

        The return value of this method can be used in control software to ensure that when moving
        towards a particular position that the direction is chosen such that it does not cross
        throught these positions. This may force the control software to take a longer path than
        it otherwise would.

        Returns:
            The encoder positions that should not be crossed. If there is no reasonable choice
            (such as for the azimuth axis of an alt-az mount) the attribute of the object can be
            set to None.
        """


    def predict(
            self,
            times_from_start: np.ndarray,
            rate_commands: np.ndarray,
            position_axis_start: float,
            slew_rate_start: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future axis positions based on a set of future commands.

        Note that the slew rate commands are assumed to be issued at the *start* of each time
        interval and the predicted positions correspond to the *end* of the same interval (which is
        also the start of the following interval). Consequently, rate_commands[0] is assumed to be
        executed immediately (0 seconds from now), followed by rate_commands[1] executed at
        times_from_start[0], etc., until the last command is issued at times_from_start[-2] and the
        final position prediction corresponds to the end of that interval at times_from_start[-1].

        Note also that Astropy objects are not used in this method for performance reasons.

        Args:
            times_from_start: An array of times in seconds measured from the starting time (t = 0).
                The returned position predictions will correspond to these instants in time.
            rate_commands: An array of slew rate commands that are assumed to be sent to the mount
                in the future. The size of this array must match the size of times_from_start. Note
                that the mount is not able to achieve these rates instantly due to acceleration
                limits. The predicted positions take this into account.
            position_axis_start: The position of the mount axis at time t = 0.
            slew_rate_start: The slew rate of the mount axis at time t = 0.

        Returns:
            A tuple where the first element is an array of predicted mount axis encoder positions
            in degrees and the second element is an array of predicted slew rates in degrees per
            second. Each entry in these arrays corresponds to elements in the times_from_start
            array.
        """
        positions_predicted = []
        rates_predicted = []
        position_current = float(position_axis_start)
        rate_at_start_of_step = float(slew_rate_start)
        time_steps = np.concatenate(((times_from_start[0],), np.diff(times_from_start)))
        slew_accel = self.slew_accel

        for time_step, rate_command in zip(time_steps, rate_commands):

            # solve for time_accel_end
            time_accel_end = np.abs(rate_command - rate_at_start_of_step) / slew_accel
            accel = slew_accel * np.sign(rate_command - rate_at_start_of_step)

            # acceleration continues to the end of this timestep
            if time_accel_end >= time_step:
                position_current += accel*time_step**2 + rate_at_start_of_step*time_step
                positions_predicted.append(position_current)
                rate_at_start_of_step += accel*time_step
                rates_predicted.append(rate_at_start_of_step)
                continue

            # no rate change this timestep
            if time_accel_end == 0:
                position_current += rate_at_start_of_step*time_step
                positions_predicted.append(position_current)
                rates_predicted.append(rate_at_start_of_step)
                continue

            # compute position at time_accel_end, the moment accel ends / commanded rate achieved
            position_at_accel_end = (accel*time_accel_end**2 + rate_at_start_of_step*time_accel_end
                                     + position_current)

            # compute position at end of timestep
            position_current = rate_command*(time_step - time_accel_end) + position_at_accel_end
            positions_predicted.append(position_current)

            rate_at_start_of_step = rate_command
            rates_predicted.append(rate_command)

        return np.array(positions_predicted) % 360, np.array(rates_predicted)


class NexStarMount(TelescopeMount):
    """Interface class to facilitate tracking with NexStar telescopes.

    This class implements the abstract methods in the TelescopeMount base class. The interface to
    the NexStar hand controller is provided by the point package.

    Attributes:
        mount: A point.NexStar object which abstracts the low-level serial command interface to the
            NexStar hand controller.
        alt_min_limit: Lower limit on the mount's altitude, which can be used to prevent the
            optical tube from colliding with the mount. Limit is enforced during calls to slew().
        alt_max_limit: Upper limit on the mount's altitude, which can be used to prevent the
            optical tube from colliding with the mount. Limit is enforced during calls to slew().
        bypass_position_limits (bool): If True altitude limits will not be enforced.
        max_slew_rate: Maximum slew rate supported by the mount in degrees per second.
        cached_position (MountEncoderPositions): The most recently queried encoder positions or
            None if get_position() has not been called since this object was constructed.
        cached_position_time (float): A Unix timestamp corresponding to the time at which the value
            in cached_position was last updated. A value of None means cached_position has never
            been populated.
    """


    class AxisName(IntEnum):
        """Mapping from axis index to/from names"""
        AZIMUTH = 0
        ALTITUDE = 1

        def short_name(self) -> str:
            """Abbreviated axis name"""
            return 'az' if self == self.AZIMUTH else 'alt'


    # pylint: disable=too-many-arguments
    def __init__(
            self,
            device_name: str,
            alt_min_limit: float = 0.0,
            alt_max_limit: float = 65.0,
            bypass_position_limits: bool = False,
            max_slew_rate: float = 16319.0/3600.0,
            slew_accel: float = 10.0,
        ):
        """Inits NexStarMount object.

        Initializes a NexStarMount object by constructing a point.NexStar object to communicate
        with the hand controller and sets initial values for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected to the hand
                controller. For example, '/dev/ttyUSB0'.
            alt_min_limit: Lower limit on the mount's altitude. The default value is reasonable for
                a NexStar 130SLT.
            alt_max_limit: Upper limit on the mount's altitude. The default value is reasonable for
                a NexStar 130SLT.
            bypass_position_limits: If True altitude limits will not be enforced.
            max_slew_rate: The maximum slew rate supported by the mount. The default value (about
                4.5 deg/s) is the max rate supported by the NexStar 130SLT hand controller as
                determined by experimentation.
            slew_accel: The approximate acceleration of the mount in degrees per second squared on
                changing slew rates. This directly affects the accuracy of `predict()` in the
                parent class.
        """
        self.mount = point.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit
        self.bypass_position_limits = bypass_position_limits
        self.max_slew_rate = max_slew_rate
        self._slew_accel = slew_accel
        self.cached_position = None
        self.cached_position_time = None
        self._rate_last_commanded = {self.AxisName.AZIMUTH: 0.0, self.AxisName.ALTITUDE: 0.0}


    @property
    def slew_accel(self) -> float:
        return self._slew_accel


    def get_position(self, max_cache_age: float = 0.0) -> MountEncoderPositions:
        """Gets the current position of the mount.

        Gets the current position coordinates of the mount. The positions returned are as reported
        by the mount with no corrections applied. The position is also cached inside this object
        for efficient altitude limit enforcement.

        Args:
            max_cache_age: If the position has been read from the mount less than this many seconds
                ago, the function may return a cached position value in lieu of reading the
                position from the mount. In cases where reading from the mount is relatively slow
                this may allow the function to return much more quickly. The default value is set
                to 0 seconds, in which case the function will never return a cached value.

        Returns:
            An instance of MountEncoderPositions where encoder_0 is the azimuth axis and encoder_1
            is the altitude axis.
        """
        if self.cached_position is not None:
            time_since_cached = time.time() - self.cached_position_time
            if time_since_cached < max_cache_age:
                return self.cached_position

        (az, alt) = self.mount.get_azalt()
        self.cached_position = MountEncoderPositions(
            Longitude(az*u.deg),
            Longitude(alt*u.deg),
        )
        self.cached_position_time = time.time()
        return self.cached_position


    def slew(self, axis: int, rate: float) -> None:
        """Command the mount to slew on one axis.

        Commands the mount to slew at a particular rate in one axis. Each axis is controlled
        independently. To slew in both axes, call this function twice: once for each axis. If the
        slew is in the altitude axis and altitude limits have been set, the function will check the
        mount's current position against the limits. If the limit has been violated and the slew
        direction is not away from the limit, the slew rate in the altitude axis will be commanded
        to 0. Note that the altitude limit protection is not guaranteed to prevent a collision of
        the optical tube against the mount for a number of reasons:
            1) The limit is only checked each time this function is called
            2) The motors do not respond instantly when commanded to stop
            3) Altitude knowledge is dependent on good alignment
            4) The limits could be set improperly
        To prevent unnecessary reads of the mount position, the altitude limit check will attempt
        to use a cached position value. However if the cached value is too old it will read the
        mount's position directly.

        Args:
            axis: The axis to affect.
            rate: The slew rate in degrees per second. The sign of the value indicates the
                direction of the slew.

        Raises:
            ValueError if the axis is out of range for the specific mount.
            ValueError if the magntiude of `rate` exceeds the max supported slew rate.
        """
        axis = self.AxisName(axis)

        if abs(rate) > self.max_slew_rate:
            raise ValueError(f'Requested rate ({rate}) exceeds max rate ({self.max_slew_rate})')

        # enforce altitude limits
        if axis == self.AxisName.ALTITUDE and not self.bypass_position_limits:
            position = self.get_position(0.25)
            if ((position[self.AxisName.ALTITUDE].deg >= self.alt_max_limit and rate > 0.0) or
                (position[self.AxisName.ALTITUDE].deg <= self.alt_min_limit and rate < 0.0)):
                rate = 0.0

        # slew_var argument units are arcseconds per second
        self.mount.slew_var(axis, rate * 3600.0)
        self._rate_last_commanded[axis] = rate


    def get_slew_rate(self, axis: int) -> float:
        """Get current slew rate of one mount axis in degrees per second.

        For this mount the actual slew rate cannot be queried directly from the mount hardware,
        therefore the best we can do is cache the last commanded rate and return that. Since the
        mount enforces acceleration limits automatically the mount may not have achieved the last
        commanded rate if this method is called soon after a large rate change.

        A more accurate measure of instantaneous slew rate may be possible by querying the mount
        position twice to calculate delta position divided by delta time.

        Args:
            axis: The axis to which this applies.

        Returns:
            The current slew rate of this mount axis in degrees per second.
        """
        return self._rate_last_commanded[axis]


    def safe(self) -> None:
        """Sets slew rates to zero on both axes. Does not block until motion has stopped."""
        logger.info('Safing mount.')
        for axis in self.AxisName:
            self.slew(axis, 0.0)


    def shutdown(self) -> None:
        """Sets slew rates to zero on both axes and disconnects. Does not block."""
        self.mount.shutdown()


    def no_cross_encoder_positions(self) -> MountEncoderPositions:
        """Indicate encoder positions that should not be crossed.

        Since this is an alt-az mount, only the altitude axis has a range of values that cannot
        be crossed. For the azimuth axis there is no particular value that should not be crossed.

        Returns:
            MountEncoderPositions: Contains the encoder positions that should not be crossed.
        """
        return MountEncoderPositions(None, Longitude(-90*u.deg, wrap_angle=180*u.deg))


class LosmandyGeminiMount(TelescopeMount):
    """Interface class for Losmandy equatorial mounts with Gemini 2.

    This class implements the abstract methods in the TelescopeMount base class. The interface to
    the Gemini 2 mount computer is provided by the point package.

    Attributes:
        mount: A point.Gemini2 object which abstracts the low-level serial or UDP command interface
            to Gemini 2.
        ra_west_limit: Right ascension west of meridian limit in degrees.
        ra_east_limit: Right ascension east of meridian limit in degrees.
        bypass_position_limits: Boolean, True when RA limits are bypassed.
        max_slew_rate: Maximum slew rate supported by the mount in degrees per second.
        cached_position (MountEncoderPositions): Cached from last time position was read from the
            mount or None if get_position() has not been called since this object was constructed.
        cached_position_time (float): Unix timestamp corresponding to the time when cached_position
            was updated or None if it has never been set.
    """


    class AxisName(IntEnum):
        """Mapping from axis index to/from names"""
        RIGHT_ASCENSION = 0
        DECLINATION = 1

        def short_name(self) -> str:
            """Axis names as used by the point package API for this mount"""
            return 'ra' if self == self.RIGHT_ASCENSION else 'dec'


    # pylint: disable=too-many-arguments
    def __init__(
            self,
            device_name: str,
            ra_west_limit: float = 110.0,
            ra_east_limit: float = 110.0,
            bypass_position_limits: bool = False,
            max_slew_rate: float = 4.0,
            slew_accel: float = 20.0,
            max_slew_step: float = 0.5,
            use_multiprocessing: bool = True,
        ):
        """Inits LosmandyGeminiMount object.

        Initializes a LosmandyGeminiMount object by constructing a point.Gemini2 object to
        communicate with Gemini 2 and sets initial values for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected to Gemini 2 or the
                host name or IP address of the Gemini network interface. For example,
                '/dev/ttyACM0' for serial or '192.168.1.1' for network connection.
            ra_west_limit: Limit right ascension axis to less than this many degrees from the
                meridian to the west.
            ra_east_limit: Limit right ascension axis to less than this many degrees from the
                meridian to the east.
            bypass_position_limits: If True RA axis limits will not be enforced.
            max_slew_rate: The maximum allowed slew rate magnitude in degrees per second.
            slew_accel: The desired acceleration of the mount in degrees per second squared when
                changing slew rates. Higher acceleration increases the likelihood of motor stalls.
            max_slew_step: The maximum change in slew rate per slew command in degrees per second.
                Higher limits increase the likelihood of motor stalls.
            use_multiprocessing: The Gemini 2 firmware as of Level 5.2 does not have very good
                support for direct control of the slew rate. The available commands try to apply
                the new slew rate instantaneously, and when the firmware detects that the encoders
                are not immediately moving at the new rate it interprets this as a stall. Setting
                this to `True` enables a work-around in the Gemini software driver in the `point`
                package which tries to smoothly accelerate the mount with a series of commands sent
                in quick succession.
        """
        self.mount = point.Gemini2(
            backend=point.gemini_backend.Gemini2BackendUDP(0.25, device_name),
            rate_limit=max_slew_rate,
            rate_step_limit=max_slew_step,
            accel_limit=slew_accel,
            use_multiprocessing=use_multiprocessing,
        )

        # If Gemini startup is not complete the coordinates it reports will not correspond to its
        # position. This can lead to bad behaviors such as tracking motion that never stops and
        # inability of this code to enforce limits on the RA axis.
        if self.mount.startup_check() != point.gemini_commands.G2StartupStatus.DONE_EQUATORIAL:
            raise RuntimeError('Gemini has not completed startup!')

        self.ra_west_limit = ra_west_limit
        self.ra_east_limit = ra_east_limit
        self.bypass_position_limits = bypass_position_limits
        self.max_slew_rate = max_slew_rate
        self._slew_accel = slew_accel
        self.cached_position = None
        self.cached_position_time = None


    @property
    def slew_accel(self) -> float:
        return self._slew_accel


    def get_position(self, max_cache_age: float = 0.0) -> MountEncoderPositions:
        """Gets the current position of the mount.

        Gets the current position coordinates of the mount. The positions returned are as reported
        by the mount with no corrections applied.

        Args:
            max_cache_age: If the position has been read from the mount less than this many seconds
                ago, the function may return a cached position value in lieu of reading the
                position from the mount. In cases where reading from the mount is relatively slow
                this may allow the function to return much more quickly. The default value is set
                to 0 seconds, in which case the function will never return a cached value.

        Returns:
            An instance of MountEncoderPositions.
        """
        if self.cached_position is not None:
            time_since_cached = time.time() - self.cached_position_time
            if time_since_cached < max_cache_age:
                return self.cached_position

        enq = self.mount.enq_macro()
        self.cached_position = MountEncoderPositions(
            Longitude(enq['pra'] * 180.0 / 1152000 * u.deg),  # motor ticks to degrees
            Longitude(enq['pdec'] * 180.0 / 1152000 * u.deg),  # motor ticks to degrees
        )
        self.cached_position_time = time.time()
        return self.cached_position


    def slew(self, axis: int, rate: float) -> None:
        """Command the mount to slew on one axis.

        Commands the mount to slew at a particular rate in one axis. Each axis is controlled
        independently. To slew in both axes, call this function twice: once for each axis.

        Args:
            axis: The axis to affect.
            rate: The slew rate in degrees per second. The sign of the value indicates the
                direction of the slew.

        Raises:
            ValueError if the axis is out of range for the specific mount.
            ValueError if the magntiude of `rate` exceeds the max supported slew rate.
        """
        axis = self.AxisName(axis)

        if abs(rate) > self.max_slew_rate:
            raise ValueError(f'Requested rate ({rate}) exceeds max rate ({self.max_slew_rate})')

        if axis == self.AxisName.RIGHT_ASCENSION and not self.bypass_position_limits:
            pra = self.get_position(0.25)[self.AxisName.RIGHT_ASCENSION.value]
            if ((pra.deg < 180 - self.ra_west_limit and rate < 0.0) or
                (pra.deg > 180 + self.ra_east_limit and rate > 0.0)):
                rate = 0.0

        self.mount.slew(axis.short_name(), rate)


    def get_slew_rate(self, axis: int) -> float:
        """Get current slew rate of one mount axis in degrees per second."""
        axis = self.AxisName(axis)
        return self.mount.get_slew_rate(axis.short_name())


    def safe(self) -> None:
        """Stop motion on both axes."""
        logger.info('Safing mount.')
        self.mount.stop_motion()


    def shutdown(self) -> None:
        """Stop motion on both axes and disconnect."""
        self.mount.shutdown()


    def no_cross_encoder_positions(self) -> MountEncoderPositions:
        """Indicate encoder positions that should not be crossed.

        At startup in the counterweight down position both encoders have a value of 180 degrees.
        The no-cross positions are exactly 180 degrees away from these startup values, at 0
        degrees.

        Returns:
            MountEncoderPositions: Contains the encoder positions that should not be crossed.
        """
        return MountEncoderPositions(Longitude(0*u.deg), Longitude(0*u.deg))


def add_program_arguments(parser: ArgParser, meridian_side_required: bool = False) -> None:
    """Add program arguments for all mounts.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
        meridian_side_required: When True, the meridian-side command line argument will be set as
            required. When False the same argument is optional.
    """
    mount_group = parser.add_argument_group(
        title='Mount Options',
        description='Options that apply to telescope mounts',
    )
    mount_group.add_argument(
        '--mount-type',
        help='select mount type (nexstar or gemini)',
        default='gemini'
    )
    mount_group.add_argument(
        '--mount-path',
        help='serial device node or hostname for mount command interface',
        default='/dev/ttyACM0'
    )
    parser.add_argument(
        '--bypass-position-limits',
        help='bypass mount axis position limits',
        action='store_true'
    )
    mount_group.add_argument(
        '--meridian-side',
        help='side of meridian for equatorial mounts to prefer',
        required=meridian_side_required,
        choices=tuple(m.name.lower() for m in MeridianSide),
    )


def make_mount_from_args(args: Namespace, use_multiprocessing: bool = True) -> TelescopeMount:
    """Construct the appropriate TelescopeMount instance based on the program arguments provided.

    Args:
        args: Set of program arguments.
        use_multiprocessing: Passed to the LosmandyGeminiMount constructor only. Ignored for other
            mount types.
    """
    if args.mount_type == 'nexstar':
        return NexStarMount(
            device_name=args.mount_path,
            bypass_position_limits=args.bypass_position_limits
        )
    elif args.mount_type == 'gemini':
        return LosmandyGeminiMount(
            device_name=args.mount_path,
            bypass_position_limits=args.bypass_position_limits,
            use_multiprocessing=use_multiprocessing,
        )
    else:
        raise ValueError(f'Invalid mount-type: {args.mount_type}')
