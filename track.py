import time
import abc

# return value limited to the range [-limit,+limit]
def clamp(x, limit):
    return max(min(limit, x), -limit)


class ErrorSource:
    __metaclass__ = abc.ABCMeta

    class NoSignalException(Exception):
        pass

    # Returns the pointing error as a dict with entries for each axis. For an
    # Az-Alt mount, the expected entries would have keys 'az' and 'alt'. The
    # error values have units of degrees. Azimuth range is [0,360) and 
    # altitude range is [-180,180). May raise a NoSignalException if the 
    # error cannot be computed.
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
    def get_azalt(self):
        pass

    # Sets the slew rate of the mount in degrees per second. May raise
    # an AltitudeLimitException if the mount altitude position is at a limit
    # and the requested altitude slew rate is not away from the limit.
    # The axis argument is a string which may take values such as 'az' and
    # 'alt' for an Az-Alt mount.
    @abc.abstractmethod
    def slew(self, axis, rate):
        pass

    # Get the current slew rate in degrees per second. For mounts that do not
    # support a slew rate readback command, this function may return a cached
    # value from the most recently commanded slew rate. The axis argument is
    # a string.
    @abc.abstractmethod
    def get_slew_rate(self, axis):
        pass

    # Returns the maximum supported slew rate in degrees per second.
    @abc.abstractmethod
    def get_max_slew_rate(self):
        pass


# Proportional plus integral (PI) loop filter
class LoopFilter:
    def __init__(self, bandwidth, damping_factor, update_period, rate_limit):
        
        # compute loop filter gains
        bt = bandwidth * update_period
        k0 = update_period
        denom = damping_factor + 1.0 / (4.0 * damping_factor)
        self.prop_gain = 4.0 * damping_factor / denom * bt / k0
        self.int_gain = 4.0 / denom**2.0 * bt**2.0 / k0

        # initialize control loop integrator
        self.int = 0.0

        self.rate_limit = rate_limit

    # Returns new slew rate in [phase units] per second, where [phase units] 
    # are the same as the units of the input error value.
    def update(self, error):
        # proportional term
        prop = self.prop_gain * error

        # integral term
        self.int = clamp(self.int + self.int_gain * error, self.rate_limit)

        # new slew rate is the sum of P and I terms subject to rate limit
        return clamp(prop + self.int, self.rate_limit)


# Main tracking loop class
class Tracker:

    def __init__(self, mount, error_source, update_period, loop_bandwidth, damping_factor):

        # update rate of control loop
        self.update_period = update_period

        self.loop_filter = {}
        self.loop_filter['az'] = LoopFilter(
            bandwidth = loop_bandwidth, 
            damping_factor = damping_factor, 
            update_period = update_period, 
            rate_limit = mount.get_max_slew_rate()
        )
        self.loop_filter['alt'] = LoopFilter(
            bandwidth = loop_bandwidth, 
            damping_factor = damping_factor, 
            update_period = update_period, 
            rate_limit = mount.get_max_slew_rate()
        )

        # object of type TelescopeMount
        self.mount = mount

        # object of type ErrorSource
        self.error_source = error_source

        self.error = {'az': None, 'alt': None}
        self.slew_rate = {'az': 0.0, 'alt': 0.0}
        self.num_iterations = 0

    # stopping condition for control loop (can override in child class)
    # returns True if tracking should stop
    def _stopping_condition(self):
        return False

    # The axes argument is a list of strings indicating which axes should be 
    # under active tracking control. The slew rate on any axis not included
    # in the list will not be commanded by the control loop.
    def run(self, axes=['az', 'alt']):

        while True:
            start_time = time.time()

            try:
                if self._stopping_condition():
                    return

                # get current pointing error
                self.error = self.error_source.compute_error()

                for axis in axes:
                    self.slew_rate[axis] = self.loop_filter[axis].update(self.error[axis])
                    self.mount.slew(axis, self.slew_rate[axis])

            except ErrorSource.NoSignalException:
                self.error['az'] = None
                self.error['alt'] = None
            except TelescopeMount.AltitudeLimitException:
                self.loop_filter['alt'].int = 0.0
            finally:
                self.num_iterations += 1

                elapsed_time = time.time() - start_time

                if elapsed_time > self.update_period:
                    print('Warning: Can''t keep up! Actual loop period this iteration: ' + str(elapsed_time))
                else:
                    time.sleep(self.update_period - elapsed_time)
