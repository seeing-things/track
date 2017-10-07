import time
import abc

# return value limited to the range [-limit,+limit]
def clamp(x, limit):
    return max(min(limit, x), -limit)


class ErrorSource(object):
    __metaclass__ = abc.ABCMeta

    class NoSignalException(Exception):
        pass

    # Returns the pointing error as a dict with entries for each axis. For an
    # Az-Alt mount, the expected entries would have keys 'az' and 'alt'. The
    # error values have units of degrees. Azimuth range is [0,360) and 
    # altitude range is [-180,180). May raise a NoSignalException if the 
    # error cannot be computed.
    @abc.abstractmethod
    def compute_error(self, retries=0):
        pass


class TelescopeMount(object):
    __metaclass__ = abc.ABCMeta

    class AltitudeLimitException(Exception):
        pass

    @abc.abstractmethod
    def get_azalt(self):
        """Gets the current position of the mount.

        Returns:
            A dict with keys 'az' and 'alt' where the values are the azimuth
            and altitude positions in degrees. The azimuth range is [0,360) and
            the altitude range is [-180,+180).
        """
        pass

    @abc.abstractmethod
    def get_aligned_slew_dir(self):
        """Gets the slew directions used during alignment.

        Returns:
            A dict with keys 'az' and 'alt' where the values are +1 or -1 
            indicating the slew direction used during alignment for that axis.
        """
        pass

    @abc.abstractmethod
    def remove_backlash(self, position, axes_to_adjust):
        """Adjusts positions to compensate for backlash deadband.

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
        pass

    @abc.abstractmethod
    def slew(self, axis, rate):
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis.

        Args:
            axis: A string indicating the axis.
            rate: A float giving the slew rate in degrees per second. The sign
                of the value indicates the direction of the slew.

        Raises:
            AltitudeLimitException: Implementation dependent.
        """
        pass

    @abc.abstractmethod
    def get_max_slew_rate(self):
        """Get the max supported slew rate.

        Returns:
            The maximum supported slew rate in degrees per second.            
        """
        pass


# Proportional plus integral (PI) loop filter
class LoopFilter(object):
    def __init__(self, bandwidth, damping_factor, rate_limit, max_update_period=1.0):
        self.bandwidth = bandwidth
        self.damping_factor = damping_factor
        self.max_update_period = max_update_period
        self.rate_limit = rate_limit
        self.int = 0.0
        self.last_iteration_time = None

    # Returns new slew rate in [phase units] per second, where [phase units] 
    # are the same as the units of the input error value.
    def update(self, error):

        # can't measure loop period on first update
        if self.last_iteration_time is None:
            self.last_iteration_time = time.time()
            return 0.0

        update_period = time.time() - self.last_iteration_time
        self.last_iteration_time = time.time()
        if update_period > self.max_update_period:
            print('Warning: loop filter update period was ' 
                + str(update_period) + ' s, limit is ' 
                + str(self.max_update_period) + ' s.')
            return self.int

        print('measured loop period: ' + str(update_period * 1000.0) + ' ms')

        # compute loop filter gains based on loop period
        bt = self.bandwidth * update_period
        k0 = update_period
        denom = self.damping_factor + 1.0 / (4.0 * self.damping_factor)
        prop_gain = 4.0 * self.damping_factor / denom * bt / k0
        int_gain = 4.0 / denom**2.0 * bt**2.0 / k0

        # proportional term
        prop = prop_gain * error

        # integral term
        self.int = clamp(self.int + int_gain * error, self.rate_limit)

        # new slew rate is the sum of P and I terms subject to rate limit
        return clamp(prop + self.int, self.rate_limit)


# Main tracking loop class
class Tracker(object):

    def __init__(self, mount, error_source, loop_bandwidth, damping_factor):

        self.loop_filter = {}
        self.loop_filter['az'] = LoopFilter(
            bandwidth = loop_bandwidth, 
            damping_factor = damping_factor, 
            rate_limit = mount.get_max_slew_rate()
        )
        self.loop_filter['alt'] = LoopFilter(
            bandwidth = loop_bandwidth, 
            damping_factor = damping_factor, 
            rate_limit = mount.get_max_slew_rate()
        )

        # object of type TelescopeMount
        self.mount = mount

        # object of type ErrorSource
        self.error_source = error_source

        self.error = {'az': None, 'alt': None}
        self.slew_rate = {'az': 0.0, 'alt': 0.0}
        self.num_iterations = 0

        self.callback = None

        self.stop = False

    # stopping condition for control loop (can override in child class)
    # returns True if tracking should stop
    def _stopping_condition(self):
        return self.stop

    # Registers a callback function to be called near the end of each loop
    # iteration. The callback function is called with no arguments. Pass
    # None as the argument value to un-register.
    def register_callback(self, callback):
        self.callback = callback

    # The axes argument is a list of strings indicating which axes should be 
    # under active tracking control. The slew rate on any axis not included
    # in the list will not be commanded by the control loop. If an empty list
    # is passed, the function returns immediately.
    def run(self, axes=['az', 'alt']):
        
        self.stop =  False

        if len(axes) == 0:
            return

        while True:
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

                if self.callback is not None:
                    self.callback()

                self.num_iterations += 1