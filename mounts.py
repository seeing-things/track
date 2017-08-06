from track import TelescopeMount
import nexstar

class NexStarMount(TelescopeMount):

    # Default value for max slew rate was determined experimentally using 
    # the NexStar 130SLT
    def __init__(self, device_name, alt_min_limit=-10.0, alt_max_limit=67.0, max_slew_rate=16319.0/3600.0):
        self.nexstar = nexstar.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit
        self.max_slew_rate = max_slew_rate
        self.cached_slew_rate = {}
        self.cached_slew_rate['az'] = 0.0
        self.cached_slew_rate['alt'] = 0.0

    def get_azalt(self):
        return self.nexstar.get_azalt()

    # Command the mount to slew at a paritcular rate in degrees per second in
    # the azimuth ('az') or altitude ('alt') axis. Throws 
    # AltitudeLimitException if slewing in the altitude axis and the 
    # altitude limits have been exceeded. When this occurs, the altitude
    # rate will be forced to zero unless the slew direction is away from the 
    # limit.
    def slew(self, axis, rate):
        assert axis in ['az', 'alt']

        # enforce altitude limits
        if axis == 'alt':
            hit_limit = False
            (mount_az_deg, mount_alt_deg) = self.get_azalt()
            if ((mount_alt_deg >= self.alt_max_limit and rate > 0.0) 
            or (mount_alt_deg <= self.alt_min_limit and rate < 0.0)):
                self.nexstar.slew_var('alt', 0.0)
                raise self.AltitudeLimitException('Altitude limit exceeded')
            
        # slew_var argument units are arcseconds per second
        self.nexstar.slew_var(axis, rate * 3600.0)

        # NexStar has no command to query the slew rate so cache the commanded
        # rate to allow get_slew_rate() method
        self.cached_slew_rate[axis] = rate

    def get_slew_rate(self, axis):
        assert axis in ['az', 'alt']
        return self.cached_slew_rate[axis]

    def get_max_slew_rate(self):
        return self.max_slew_rate
