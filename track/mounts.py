"""mounts for use in telescope tracking control loop.

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

from abc import ABC, abstractmethod
import enum
from enum import IntEnum
from typing import NamedTuple, Tuple
import time
import numpy as np
from astropy import units as u
from astropy.coordinates import Longitude
import point


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
    def slew(self, axis: int, rate: float) -> Tuple[float, bool]:
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

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to quantization or due to enforcement of slew rate, acceleration, or axis position
                limits. The second element is a boolean that is set to True if any of the limits
                enforced by the mount were exceeded by the requested slew rate.

        Raises:
            ValueError if the axis is out of range for the specific mount.
        """


    @abstractmethod
    def safe(self) -> bool:
        """Bring mount into a safe state.

        This method will do whatever is necessary to bring the mount into a safe state, such that
        there is no risk to hardware if the program terminates immediately afterward and
        communication with the mount stops. At minimum this method will stop all motion.

        Returns:
            True if the mount was safed successfully. False otherwise.
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
        bypass_alt_limits (bool): If True altitude limits will not be enforced.
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
            bypass_alt_limits: bool = False,
            max_slew_rate: float = 16319.0/3600.0,
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
            bypass_alt_limits: If True altitude limits will not be enforced.
            max_slew_rate: The maximum slew rate supported by the mount. The default value (about
                4.5 deg/s) is the max rate supported by the NexStar 130SLT hand controller as
                determined by experimentation.
        """
        self.mount = point.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit
        self.bypass_alt_limits = bypass_alt_limits
        self.max_slew_rate = max_slew_rate
        self.cached_position = None
        self.cached_position_time = None


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


    def slew(self, axis: int, rate: float) -> Tuple[float, bool]:
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis. Each axis is controlled
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
            axis (AxisName): The axis to affect.
            rate (float): The slew rate in degrees per second. The sign of the value indicates the
                direction of the slew.

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to slew rate or axis position limits. The second element is a boolean that is set
                to True if any of the limits enforced by the mount were exceeded by the requested
                slew rate.
        """
        axis = self.AxisName(axis)

        limits_exceeded = False
        if abs(rate) > self.max_slew_rate:
            limits_exceeded = True
            rate = np.clip(rate, -self.max_slew_rate, self.max_slew_rate)

        # enforce altitude limits
        if axis == self.AxisName.ALTITUDE and not self.bypass_alt_limits:
            position = self.get_position(0.25)
            # pylint: disable=bad-continuation
            if ((position[self.AxisName.ALTITUDE].deg >= self.alt_max_limit and rate > 0.0) or
                (position[self.AxisName.ALTITUDE].deg <= self.alt_min_limit and rate < 0.0)):
                limits_exceeded = True
                rate = 0.0

        # slew_var argument units are arcseconds per second
        self.mount.slew_var(axis, rate * 3600.0)

        return (rate, limits_exceeded)


    def safe(self) -> bool:
        """Bring mount into a safe state.

        For this mount the only action necessary is to command the slew rate to zero on both axes.

        Returns:
            True if the mount was safed successfully. False otherwise.
        """
        success = True
        for axis in self.AxisName:
            rate, _ = self.slew(axis, 0.0)
            if rate != 0.0:
                success = False

        return success


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
        bypass_ra_limits: Boolean, True when RA limits are bypassed.
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
            bypass_ra_limits: bool = False,
            max_slew_rate: float = 4.0,
            max_slew_accel: float = 40.0,
            max_slew_step: float = 0.5,
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
            bypass_ra_limits: If True RA axis limits will not be enforced.
            max_slew_rate: The maximum allowed slew rate magnitude in degrees per second.
            max_slew_accel: The maximum allowed slew acceleration in degrees per second squared.
                Higher limits increase the likelihood of motor stalls.
            max_slew_step: The maximum change in slew rate per slew command in degrees per second.
                Higher limits increase the likelihood of motor stalls.
        """
        self.mount = point.Gemini2(
            backend=point.gemini_backend.Gemini2BackendUDP(0.25, device_name),
            rate_limit=max_slew_rate,
            rate_step_limit=max_slew_step,
            accel_limit=max_slew_accel,
            use_multiprocessing=True,
        )

        # If Gemini startup is not complete the coordinates it reports will not correspond to its
        # position. This can lead to bad behaviors such as tracking motion that never stops and
        # inability of this code to enforce limits on the RA axis.
        if self.mount.startup_check() != point.gemini_commands.G2StartupStatus.DONE_EQUATORIAL:
            raise RuntimeError('Gemini has not completed startup!')

        self.ra_west_limit = ra_west_limit
        self.ra_east_limit = ra_east_limit
        self.bypass_ra_limits = bypass_ra_limits
        self.max_slew_rate = max_slew_rate
        self.cached_position = None
        self.cached_position_time = None


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

        enq, time_resp = self.mount.enq_macro()
        self.cached_position = MountEncoderPositions(
            Longitude(enq['pra'] * 180.0 / 1152000 * u.deg),  # motor ticks to degrees
            Longitude(enq['pdec'] * 180.0 / 1152000 * u.deg),  # motor ticks to degrees
        )
        self.cached_position_time = time.time()
        return self.cached_position, time_resp


    def slew(self, axis: int, rate: float) -> Tuple[float, bool]:
        """Command the mount to slew on one axis.

        Commands the mount to slew at a particular rate in one axis. Each axis is controlled
        independently. To slew in both axes, call this function twice: once for each axis.

        Args:
            axis (AxisName): The axis to affect.
            rate (float): The slew rate in degrees per second. The sign of the value indicates the
                direction of the slew.

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to quantization or due to enforcement of slew rate, acceleration, or axis position
                limits. The second element is a boolean that is set to True if any of the limits
                enforced by the mount were exceeded by the requested slew rate.

        Raises:
            ValueError: If an invalid axis is requested.
        """
        axis = self.AxisName(axis)

        axis_limit_exceeded = False
        if axis == self.AxisName.RIGHT_ASCENSION and not self.bypass_ra_limits:
            pra = self.get_position(0.25)[self.AxisName.RIGHT_ASCENSION.value]
            # pylint: disable=bad-continuation
            if ((pra.deg < 180 - self.ra_west_limit and rate < 0.0) or
                (pra.deg > 180 + self.ra_east_limit and rate > 0.0)):
                axis_limit_exceeded = True
                rate = 0.0

        (actual_rate, limits_exceeded) = self.mount.slew(axis.short_name(), rate)

        if axis_limit_exceeded:
            limits_exceeded = True

        return (actual_rate, limits_exceeded)


    def safe(self) -> bool:
        """Bring mount into a safe state by stopping motion.

        This method blocks until motion has ceased.

        Returns:
            True when motion is stopped. Will not return otherwise.
        """
        # This method blocks until motion on both axes has ceased.
        self.mount.stop_motion()
        return True


    def no_cross_encoder_positions(self) -> MountEncoderPositions:
        """Indicate encoder positions that should not be crossed.

        At startup in the counterweight down position both encoders have a value of 180 degrees.
        The no-cross positions are exactly 180 degrees away from these startup values, at 0
        degrees.

        Returns:
            MountEncoderPositions: Contains the encoder positions that should not be crossed.
        """
        return MountEncoderPositions(Longitude(0*u.deg), Longitude(0*u.deg))
