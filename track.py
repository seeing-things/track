import ephem
import time
import datetime
import math
import nexstar
import threading
import abc

# return value limited to the range [-limit,+limit]
def clamp(x, limit):
    return max(min(limit, x), -limit)

# wraps a phase angle in degrees to the range [-180,+180)
def wrap_error(e):
    return (e + 180.0) % 360.0 - 180.0


class ErrorSource:
    __metaclass__ = abc.ABCMeta

    # returns the pointing error as a tuple of (az, alt) in degrees where the 
    # azimuth range is [0,360) and the altitude range is [-180,180)
    @abc.abstractmethod
    def compute_error(self):
        pass


class TelescopeMount:
    __metaclass__ = abc.ABCMeta

    class AltitudeLimitException(Exception):
        pass

    # Returns the current position of the mount as a tuple containing
    # (azimuth, altitude) in degrees.
    @abc.abstractmethod
    def get_azel(self):
        pass

    # Sets the slew rate of the mount in degrees per second. May raise
    # an AltitudeLimitException if the mount altitude position is at a limit
    # and the requested altitude slew rate is not away from the limit.
    @abc.abstractmethod
    def slew(self, rate_az, rate_alt):
        pass

    # Returns the maximum supported slew rate in degrees per second
    @abc.abstractmethod
    def get_max_slew_rate(self):
        pass


class NexStarMount(TelescopeMount):

    def __init__(self, device_name, alt_min_limit=-10.0, alt_max_limit=67.0):
        self.nexstar = nexstar.NexStar(device_name)
        self.alt_min_limit = alt_min_limit
        self.alt_max_limit = alt_max_limit

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

        if hit_limit:
            raise self.AltitudeLimitException('Altitude limit exceeded')

    # This value was determined experimentally using the NexStar 130SLT
    def get_max_slew_rate(self):
        return 16319.0 / 3600.0


class BlindErrorSource(ErrorSource):

    def __init__(self, mount, observer, target):

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # TelescopeMount object
        self.mount = mount

    def compute_error(self):

        # get current coordinates of the target in degrees
        # not using ephem.now() because it rounds time to the nearest second
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_az_deg = self.target.az * 180.0 / math.pi
        target_alt_deg = self.target.alt * 180.0 / math.pi

        # get current position of telescope (degrees)
        (scope_az_deg, scope_alt_deg) = self.mount.get_azel()
         
        # compute pointing errors in degrees
        error_az = wrap_error(target_az_deg - scope_az_deg)
        error_alt = wrap_error(target_alt_deg - scope_alt_deg)

        return (error_az, error_alt)


# Two independent proportional plus integral (PI) loop filters, one for 
# azimuth and another for altitude.
class LoopFilter:
    def __init__(self, bandwidth, damping_factor, update_period, rate_limit):
        
        # compute loop filter gains
        bt = bandwidth * update_period
        k0 = update_period
        denom = damping_factor + 1.0 / (4.0 * damping_factor)
        self.prop_gain = 4.0 * damping_factor / denom * bt / k0
        self.int_gain = 4.0 / denom**2.0 * bt**2.0 / k0

        # init control loop integrators
        self.int_az = 0.0
        self.int_alt = 0.0

        self.rate_limit = rate_limit

    # returns new slew rates as a tuple (slew_az, slew_alt) in [phase units] 
    # per second, where [phase units] are the same as the units of the input
    # error values
    def update(self, error_az, error_alt):
        # proportional term
        prop_az = self.prop_gain * error_az
        prop_alt = self.prop_gain * error_alt

        # integral term
        self.int_az = clamp(self.int_az + self.int_gain * error_az, self.rate_limit)
        self.int_alt = clamp(self.int_alt + self.int_gain * error_alt, self.rate_limit)

        # output is the sum of proportional and integral terms subject to rate limit
        slew_rate_az = clamp(prop_az + self.int_az, self.rate_limit)
        slew_rate_alt = clamp(prop_alt + self.int_alt, self.rate_limit)
        return (slew_rate_az, slew_rate_alt)


# Main tracking loop class
class Tracker:

    def __init__(self, mount, error_source, update_period, loop_bandwidth, damping_factor):

        # update rate of control loop
        self.update_period = update_period

        self.loop_filter = LoopFilter(
            bandwidth = loop_bandwidth, 
            damping_factor = damping_factor, 
            update_period = update_period, 
            rate_limit = mount.get_max_slew_rate()
        )

        # object of type TelescopeMount
        self.mount = mount

        # object of type ErrorSource
        self.error_source = error_source


    def start(self):
        self.start_time = time.time()
        self.running = True
        self.do_iteration()

    def stop(self):
        self.running = False

    def do_iteration(self):
        if self.running:
            threading.Timer(self.update_period, self.do_iteration).start()
        else:
            return
        
        try:
            elapsed_time = time.time() - self.start_time

            # get current pointing error
            (error_az, error_alt) = self.error_source.compute_error()

            # loop filter -- outputs are new slew rates in degrees/second
            (slew_rate_az, slew_rate_alt) = self.loop_filter.update(error_az, error_alt)

            # update mount slew rates
            try:
                self.mount.slew(slew_rate_az, slew_rate_alt)
            except self.mount.AltitudeLimitException:
                self.loop_filter.int_alt = 0.0
        
        except:
            self.stop()
            raise
