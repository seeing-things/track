from .control import TelescopeMount
import point
import time

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
        last_slew_dir: A dict storing the directions of the last commanded
            slews in each axis. Keys are 'az' and 'alt'. Values are +1 when
            the last slew was in the positive direction and -1 when the last
            slew was in the negative direction. The values may be either +1
            or -1 after class construction or when the last slew has rate 0.
        backlash: A dict storing the magnitudes of the backlash in each axis.
            Keys are 'az' and 'alt'. The values have units of degrees and are
            non-negative.
        aligned_slew_dir: A dict storing the final approach direction used
            during alignment on each axis. Keys are 'az' and 'alt'. Values
            are +1 to indicate that the final approach slew was in the positive
            direction or -1 to indicate the opposite.
    """

    def __init__(
        self,
        device_name,
        alt_min_limit=0.0,
        alt_max_limit=65.0,
        bypass_alt_limits=False,
        max_slew_rate=16319.0/3600.0,
        max_slew_accel=None,
    ):
        """Inits NexStarMount object.

        Initializes a NexStarMount object by constructing a point.NexStar
        object to communicate with the hand controller and sets initial values
        for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected
                to the hand controller. For example, '/dev/ttyUSB0'.
            alt_min_limit: Lower limit on the mount's altitude. The default
                value is reasonable for a NexStar 130SLT.
            alt_max_limit: Upper limit on the mount's altitude. The default
                value is reasonable for a NexStar 130SLT.
            bypass_alt_limits: If True altitude limits will not be enforced.
            max_slew_rate: The maximum slew rate supported by the mount. The
                default value (about 4.5 deg/s) is the max rate supported by
                the NexStar 130SLT hand controller as determined by
                experimentation.
            max_slew_accel: The maximum slew acceleration in degrees per second
                squared. The NexStar 130SLT mount has very slow acceleration
                and never stalls so there is little risk of leaving this set
                to None (no limit).
        """
        self.mount = point.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit
        self.bypass_alt_limits = bypass_alt_limits
        self.max_slew_rate = max_slew_rate
        self.max_slew_accel = max_slew_accel
        self.backlash = {'az': 0.0, 'alt': 0.0}
        self.aligned_slew_dir = {'az': +1, 'alt': +1}
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
            the altitude range is [-180,+180).
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

    def backlash_supported(self):
        return True

    def get_aligned_slew_dir(self):
        """Gets the slew directions used during alignment.

        Returns:
            A dict with keys 'az' and 'alt' where the values are +1 or -1
            indicating the slew direction used during alignment for that axis.
        """
        return self.aligned_slew_dir

    def remove_backlash(self, position, axes_to_adjust):
        """Adjusts positions to compensate for backlash deadband.

        The position in a given axis will be corrected to remove the backlash
        deadband if the set_backlash function has been called with a non-zero
        correction factor and the axis appears in the axes_to_compensate list.
        It is the responsibility of the caller to decide when backlash
        compensation should be applied. Generally compensation should only be
        applied when the mount is slewing against the direction used during
        alignment. No attempt is made to handle the special case where the mount
        is within the deadband region.

        Args:
            position: A dict with keys 'az' and 'alt' with values corresponding
                to the azimuth and altitude positions in degrees to be
                corrected.
            axes_to_adjust: A dict with keys 'az' and 'alt' and values True
                or False indicating which axes should be compensated.

        Returns:
            A dict with keys 'az' and 'alt' where the values are the azimuth
            and altitude positions in degrees with corrections applied. The
            azimuth range is [0,360) and the altitude range is [-180,+180).
        """
        az = position['az']
        alt = position['alt']

        if axes_to_adjust['az']:
            az += self.backlash['az'] * self.aligned_slew_dir['az']
            az = az % 360.0

        if axes_to_adjust['alt']:
            alt += self.backlash['alt'] * self.aligned_slew_dir['alt']
            alt = (alt + 180.0) % 360.0 - 180.0

        return {'az': az, 'alt': alt}

    def set_backlash(self, axis, aligned_slew_dir, backlash):
        """Sets the backlash compensation in one axis.

        Sets the magnitude of the backlash compensation to be applied on a
        specified axis and sets the direction in which the compensation is to
        be applied.

        Args:
            axis: A string indicating the axis: 'az' or 'alt'.
            aligned_slew_dir: An integer indicating which slew direction was
                used on final approach during alignment. May be either +1,
                indicating that final approach was in the positive direction,
                or -1 to indicate the opposite. The backlash compensation will
                only be applied when the slew direction is opposite of the
                approach direction used during alignment.
            backlash: The non-negative magnitude of the backlash adjustment in
                degrees.
        """
        assert axis in ['az', 'alt']
        assert aligned_slew_dir in [-1, +1]
        assert backlash >= 0.0
        self.backlash[axis] = backlash
        self.aligned_slew_dir[axis] = aligned_slew_dir

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

        Raises:
            AxisLimitException: Altitude limit has been exceeded and the
                altitude slew rate is not away from the limit.
        """
        assert axis in ['az', 'alt']
        if abs(rate) > self.max_slew_rate:
            raise ValueError('slew rate exceeds limit')

        # enforce altitude limits
        if axis == 'alt' and self.bypass_alt_limits == False:
            position = self.get_azalt(0.25)
            if ((position['alt'] >= self.alt_max_limit and rate > 0.0) or
                (position['alt'] <= self.alt_min_limit and rate < 0.0)):
                self.mount.slew_var('alt', 0.0)
                raise self.AxisLimitException(['alt'])

        # slew_var argument units are arcseconds per second
        self.mount.slew_var(axis, rate * 3600.0)

    def get_max_slew_rates(self):
        return {'az': self.max_slew_rate, 'alt': self.max_slew_rate}

    def get_max_slew_accels(self):
        return {'ra': self.max_slew_accel, 'dec': self.max_slew_accel}


class LosmandyGeminiMount(TelescopeMount):
    """Interface class for Losmandy equatorial mounts with Gemini 2.

    This class implements the abstract methods in the TelescopeMount base
    class. The interface to the Gemini 2 mount computer is provided by the
    point package.

    Attributes:
        mount: A point.Gemini2 object which abstracts the low-level serial or
            UDP command interface to Gemini 2.
        ra_west_limit: Right ascension west of meridian limit in degrees.
        ra_east_limit: Right ascension east of meridian limit in degrees.
        bypass_ra_limits: Boolean, True when RA limits are bypassed.
        max_slew_rate: Maximum slew rate supported by the mount in degrees per
            second.
        max_slew_accel: Maximum slew acceleration in degrees per second squared.
        cached_position: Cached position dict from last time position was read
            from the mount.
        cached_position_time: Unix timestamp corresponding to the time when
            cached_position was read from the mount.
    """

    def __init__(
        self,
        device_name,
        ra_west_limit=110.0,
        ra_east_limit=110.0,
        bypass_ra_limits=False,
        max_slew_rate=4.0,
        max_slew_accel=10.0,
    ):
        """Inits LosmandyGeminiMount object.

        Initializes a LosmandyGeminiMount object by constructing a
        point.Gemini2 object to communicate with Gemini 2 and sets initial
        svalues for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected
                to Gemini 2. For example, '/dev/ttyACM0'.
            ra_west_limit: Limit right ascension axis to less than this many
                degrees from the meridian to the west.
            ra_east_limit: Limit right ascension axis to less than this many
                degrees from the meridian to the east.
            bypass_ra_limits: If True RA axis limits will not be enforced.
            max_slew_rate: The maximum slew rate supported by the mount in
                degrees per second.
            max_slew_accel: The maximum slew acceleration supported by the
                mount in degrees per second squared. Higher limits increase
                the likelihood of motor stalls.
        """
        self.mount = point.Gemini2(device_name)
        self.ra_west_limit = ra_west_limit
        self.ra_east_limit = ra_east_limit
        self.bypass_ra_limits = bypass_ra_limits
        self.max_slew_rate = max_slew_rate
        self.max_slew_accel = max_slew_accel
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

    def backlash_supported(self):
        return False

    def get_aligned_slew_dir(self):
        raise RuntimeError('not supported')

    def remove_backlash(self, position, axes_to_adjust):
        raise RuntimeError('not supported')

    def slew(self, axis, rate):
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis. Each axis
        is controlled independently. To slew in both axes, call this function
        twice: once for each axis.

        Args:
            axis: A string indicating the axis: 'ra' or 'dec'.
            rate: A float giving the slew rate in degrees per second. The sign
                of the value indicates the direction of the slew.

        Raises:
            ValueError: If axis name is not 'ra' or 'dec'.
            AxisLimitException: RA limit has been exceeded and the RA slew rate
                is not away from the limit.
        """
        if axis == 'ra':
            if not self.bypass_ra_limits:
                pra = self.get_position()['pra']
                if ((pra < 180 - self.ra_west_limit and rate < 0.0) or
                    (pra > 180 + self.ra_east_limit and rate > 0.0)):
                    self.mount.slew_ra(0.0)
                    raise self.AxisLimitException(['ra'])
            self.mount.slew_ra(rate)
        elif axis == 'dec':
            self.mount.slew_dec(rate)
        else:
            raise ValueError("axis must be 'ra' or 'dec'")

    def get_max_slew_rates(self):
        return {'ra': self.max_slew_rate, 'dec': self.max_slew_rate}

    def get_max_slew_accels(self):
        return {'ra': self.max_slew_accel, 'dec': self.max_slew_accel}
