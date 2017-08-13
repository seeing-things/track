from track import TelescopeMount
import nexstar

class NexStarMount(TelescopeMount):
    """Interface class to facilitate tracking with NexStar telescopes.

    This class implements the abstract methods in the TelescopeMount base
    class. The interface to the NexStar hand controller is provided by
    the nexstar package.

    Attributes:
        nexstar: A nexstar.NexStar object which abstracts the low-level serial
            command interface to the NexStar hand controller.
        alt_min_limit: Lower limit on the mount's altitude, which can be used
            to prevent the optical tube from colliding with the mount. Limit
            is enforced during calls to slew().
        alt_max_limit: Upper limit on the mount's altitude, which can be used
            to prevent the optical tube from colliding with the mount. Limit
            is enforced during calls to slew().
        max_slew_rate: Maximum slew rate supported by the mount in degrees per
            second.
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

    def __init__(self, device_name, alt_min_limit=-10.0, alt_max_limit=67.0, max_slew_rate=16319.0/3600.0):
        """Inits NexStarMount object.

        Initializes a NexStarMount object by constructing a nexstar.NexStar
        object to communicate with the hand controller and sets initial values
        for several class attributes.

        Args:
            device_name: A string with the name of the serial device connected
                to the hand controller. For example, '/dev/ttyUSB0'.
            alt_min_limit: Lower limit on the mount's altitude. The default 
                value is reasonable for a NexStar 130SLT.
            alt_max_limit: Upper limit on the mount's altitude. The default
                value is reasonable for a NexStar 130SLT.
            max_slew_rate: The maximum slew rate supported by the mount. The
                default value (about 4.5 deg/s) is the max rate supported by
                the NexStar 130SLT hand controller as determined by
                experimentation.
        """
        self.nexstar = nexstar.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit
        self.max_slew_rate = max_slew_rate
        self.last_slew_dir = {'az': +1, 'alt': +1}
        self.backlash = {'az': 0.0, 'alt': 0.0}
        self.aligned_slew_dir = {'az': +1, 'alt': +1}

    def get_azalt(self, remove_backlash=True):
        """Gets the current position of the mount.

        Gets the current position coordinates of the mount in azimuth-altitude
        format. The values reported will be corrected to account for backlash
        if the set_backlash function has been called with non-zero correction
        factors and remove_backlash is set to True. The correction value is 
        added or subtracted from the raw mount-reported position when the 
        last commanded slew direction is opposite of the final approach 
        direction used during alignment. Note that this correction algorithm
        does not attempt to handle the special case where the mount drive is 
        in the deadband region.

        Args:
            remove_backlash: When true, backlash compensation is applied.
                Otherwise the uncorrected position is returned.

        Returns:
            A dict with keys 'az' and 'alt' where the values are the azimuth
            and altitude positions in degrees. The azimuth range is [0,360) and
            the altitude range is [-180,+180).
        """
        (az, alt) = self.nexstar.get_azalt()
        
        if remove_backlash:
            if self.last_slew_dir['az'] != self.aligned_slew_dir['az']:
                az += self.backlash['az'] * self.aligned_slew_dir['az']
                az = az % 360.0

            if self.last_slew_dir['alt'] != self.aligned_slew_dir['alt']:
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
        to prevent a collision of the optical tube against the mount because
        the position of the mount is only checked when this function is called.

        Args:
            axis: A string indicating the axis: 'az' or 'alt'.
            rate: A float giving the slew rate in degrees per second. The sign
                of the value indicates the direction of the slew.

        Raises:
            AltitudeLimitException: Altitude limit has been exceeded and the
                altitude slew rate has been set to zero.
        """
        assert axis in ['az', 'alt']
        assert abs(rate) <= self.max_slew_rate

        # enforce altitude limits
        if axis == 'alt':
            hit_limit = False
            position = self.get_azalt()
            if ((position['alt'] >= self.alt_max_limit and rate > 0.0) or
                (position['alt'] <= self.alt_min_limit and rate < 0.0)):
                self.nexstar.slew_var('alt', 0.0)
                raise self.AltitudeLimitException('Altitude limit exceeded')
            
        # slew_var argument units are arcseconds per second
        self.nexstar.slew_var(axis, rate * 3600.0)

        # cached to allow backlash compensation in get_azalt()
        self.last_slew_dir[axis] = +1 if rate >= 0.0 else -1

    def get_max_slew_rate(self):
        return self.max_slew_rate
