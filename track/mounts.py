"""mounts for use in telescope tracking control loop.

A set of classes that inherit from the abstract base class TelescopeMount (defined in control.py).
Each class is a wrapper for a lower level interface to the mount, creating a common interface that
can be incorporated into the Tracker class. The abstraction is not perfect and some mounts require
special interfaces that go beyond what is required by TelescopeMount.
"""

import time
import point
from .control import TelescopeMount

class NexStarMount(TelescopeMount):
    """Interface class to facilitate tracking with NexStar telescopes.

    This class implements the abstract methods in the TelescopeMount base
    class. The interface to the NexStar hand controller is provided by
    the point package.

    Attributes:
        mount: A point.NexStar object which abstracts the low-level serial
            command interface to the NexStar hand controller.
        alt_min_limit: Lower limit on the mount's altitude, which can be used
            to prevent the optical tube from colliding with the mount. Limit
            is enforced during calls to slew().
        alt_max_limit: Upper limit on the mount's altitude, which can be used
            to prevent the optical tube from colliding with the mount. Limit
            is enforced during calls to slew().
        max_slew_rate: Maximum slew rate supported by the mount in degrees per
            second.
        max_slew_accel: Maximum slew acceleration in degrees per second squared.
    """

    def __init__(
            self,
            device_name,
            alt_min_limit=0.0,
            alt_max_limit=65.0,
            bypass_alt_limits=False,
            max_slew_rate=16319.0/3600.0,
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

    def get_position(self, max_cache_age=0.0):
        """Gets the current position of the mount.

        Gets the current position coordinates of the mount in azimuth-altitude
        format. The positions returned are as reported by the mount with no
        corrections applied. The position is also cached inside this object for
        efficient altitude limit enforcement.

        Args:
            max_cache_age: If the position has been read from the mount less than this many seconds
                ago, the function may return a cached position value in lieu of reading the
                position from the mount. In cases where reading from the mount is relatively slow
                this may allow the function to return much more quickly. The default value is set
                to 0 seconds, in which case the function will never return a cached value.

        Returns:
            A dict with keys 'az' and 'alt' where the values are the azimuth
            and altitude positions in degrees. The azimuth range is [0,360) and
            the altitude range is [-90,+90].
        """
        if self.cached_position is not None:
            time_since_cached = time.time() - self.cached_position_time
            if time_since_cached < max_cache_age:
                return self.cached_position

        (az, alt) = self.mount.get_azalt()
        self.cached_position = {'az': az, 'alt': alt}
        self.cached_position_time = time.time()
        return self.cached_position

    def get_axis_names(self):
        return ['az', 'alt']

    def slew(self, axis, rate):
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis. Each axis
        is controlled independently. To slew in both axes, call this function
        twice: once for each axis. If the slew is in the altitude axis and
        altitude limits have been set, the function will check the mount's
        current position against the limits. If the limit has been violated
        and the slew direction is not away from the limit, the slew rate in
        the altitude axis will be set to 0 and an AltitudeLimitException will
        be raised. Note that the altitude limit protection is not guaranteed
        to prevent a collision of the optical tube against the mount for a
        number of reasons:
            1) The limit is only checked each time this function is called
            2) The motors do not respond instantly when commanded to stop
            3) Altitude knowledge is dependent on good alignment
            4) The limits could be set improperly
        To prevent unnecessary reads of the mount position, the altitude limit
        check will attempt to use a cached position value. However if the
        cached value is too old it will read the mount's position directly.

        Args:
            axis: A string indicating the axis: 'az' or 'alt'.
            rate: A float giving the slew rate in degrees per second. The sign
                of the value indicates the direction of the slew.

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to slew rate or axis position limits. The second element is a boolean that is set
                to True if any of the limits enforced by the mount were exceeded by the requested
                slew rate.
        """
        assert axis in ['az', 'alt']

        limit_exceeded = False
        if abs(rate) > self.max_slew_rate:
            limits_exceeded = True
            rate = clamp(rate, self.max_slew_rate)

        # enforce altitude limits
        if axis == 'alt' and not self.bypass_alt_limits:
            position = self.get_position(0.25)
            if ((position['alt'] >= self.alt_max_limit and rate > 0.0) or
                (position['alt'] <= self.alt_min_limit and rate < 0.0)):
                limits_exceeded = True
                rate = 0.0

        # slew_var argument units are arcseconds per second
        self.mount.slew_var(axis, rate * 3600.0)

        return (rate, limits_exceeded)


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
        cached_position: Cached position dict from last time position was read from the mount.
        cached_position_time: Unix timestamp corresponding to the time when cached_position was
            read from the mount.
    """

    def __init__(
            self,
            device_name,
            ra_west_limit=110.0,
            ra_east_limit=110.0,
            bypass_ra_limits=False,
            max_slew_rate=4.0,
            max_slew_accel=10.0,
            max_slew_step=0.4,
        ):
        """Inits LosmandyGeminiMount object.

        Initializes a LosmandyGeminiMount object by constructing a point.Gemini2 object to
        communicate with Gemini 2 and sets initial values for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected to Gemini 2. For
                example, '/dev/ttyACM0'.
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

    def get_axis_names(self):
        return ['ra', 'dec']

    def get_position(self, max_cache_age=0.0):
        """Gets the current position of the mount.

        Gets the current position coordinates of the mount in azimuth-altitude
        format. The positions returned are as reported by the mount with no
        corrections applied.

        Args:
            max_cache_age: If the position has been read from the mount less than this many seconds
                ago, the function may return a cached position value in lieu of reading the
                position from the mount. In cases where reading from the mount is relatively slow
                this may allow the function to return much more quickly. The default value is set
                to 0 seconds, in which case the function will never return a cached value.

        Returns:
            A dict with the following keys:
                'ra': right ascension [0, 360)
                'dec': declination [-90, +90]
                'ha': hour angle [-180, 180)
                'pra': physical right ascension position [0, 360)
                'pdec': physical declination position [0, 360)
        """
        if self.cached_position is not None:
            time_since_cached = time.time() - self.cached_position_time
            if time_since_cached < max_cache_age:
                return self.cached_position

        enq = self.mount.enq_macro()
        self.cached_position = {
            'ra': enq['ra'] * 360.0 / 24.0, # hours to degrees
            'dec': enq['dec'],
            'ha': enq['ha'] * 360.0 / 24.0, # hours to degrees
            'pra': enq['pra'] * 180.0 / 1152000, # motor ticks to degrees
            'pdec': enq['pdec'] * 180.0 / 1152000, # motor ticks to degrees
        }
        self.cached_position_time = time.time()
        return self.cached_position

    def slew(self, axis, rate):
        """Command the mount to slew on one axis.

        Commands the mount to slew at a particular rate in one axis. Each axis is controlled
        independently. To slew in both axes, call this function twice: once for each axis.

        Args:
            axis: A string indicating the axis: 'ra' or 'dec'.
            rate: A float giving the slew rate in degrees per second. The sign of the value
                indicates the direction of the slew.

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to quantization or due to enforcement of slew rate, acceleration, or axis position
                limits. The second element is a boolean that is set to True if any of the limits
                enforced by the mount were exceeded by the requested slew rate.

        Raises:
            ValueError: If axis name is not 'ra' or 'dec'.
        """
        if axis not in ['ra', 'dec']:
            raise ValueError("axis must be 'ra' or 'dec'")

        axis_limit_exceeded = False
        if axis == 'ra' and not self.bypass_ra_limits:
            pra = self.get_position()['pra']
            if ((pra < 180 - self.ra_west_limit and rate < 0.0) or
                (pra > 180 + self.ra_east_limit and rate > 0.0)):
                axis_limit_exceeded = True
                rate = 0.0

        (actual_rate, limits_exceeded) = self.mount.slew(axis, rate)

        if axis_limit_exceeded:
            limits_exceeded = True

        return (actual_rate, limits_exceeded)
