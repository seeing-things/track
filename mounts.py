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
        self.slew_rate_az = 0
        self.slew_rate_alt = 0

    def get_azel(self):
        return self.nexstar.get_azel()

    def slew(self, rate_az, rate_alt):

        # enforce altitude limits
        hit_limit = False
        (mount_az_deg, mount_alt_deg) = self.get_azel()
        if mount_alt_deg >= self.alt_max_limit and rate_alt > 0.0:
            rate_alt = 0.0
            hit_limit = True
        elif mount_alt_deg <= self.alt_min_limit and rate_alt < 0.0:
            rate_alt = 0.0
            hit_limit = True

        # slew_var argument units are arcseconds per second
        self.nexstar.slew_var(rate_az * 3600.0, rate_alt * 3600.0)

        # NexStar has no command to query the slew rate so cache the commanded
        # rates to allow get_slew_rate() method
        self.slew_rate_az = rate_az
        self.slew_rate_alt = rate_alt

        if hit_limit:
            raise self.AltitudeLimitException('Altitude limit exceeded')

        return (rate_az, rate_alt)

    def get_slew_rate(self):
        return (self.slew_rate_az, self.slew_rate_alt)

    def get_max_slew_rate(self):
        return self.max_slew_rate
