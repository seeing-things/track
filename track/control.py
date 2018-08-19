"""tracking loop for telescope control.

Track provides the classes required to point a telescope with software using a
feedback control loop.

"""

from __future__ import print_function
import time
import abc
from track.telem import TelemSource
from track.mathutils import clamp


class ErrorSource(object):
    """Abstract parent class for error sources.

    This class provides some abstract methods to provide a common interface
    for error sources to be used in the tracking loop. All error sources must
    inheret from this class and implement the methods defined.
    """
    __metaclass__ = abc.ABCMeta

    class NoSignalException(Exception):
        """Raised when no signal is available for error calculation."""
        pass

    @abc.abstractmethod
    def get_axis_names(self):
        """Get axis names

        Returns:
            A list of strings giving abbreviated names of each axis. These must
            be the keys of the dict returned by the compute_error method.
        """
        pass

    @abc.abstractmethod
    def compute_error(self):
        """Computes the error signal.

        Returns:
            The pointing error as a dict with entries for each axis. The units
            should be degrees.

        Raises:
            NoSignalException: If the error cannot be computed.
        """
        pass


class TelescopeMount(object):
    """Abstract parent class for telescope mounts.

    This class provides some abstract methods to provide a common interface
    for telescope mounts to be used in the tracking loop. All mounts must
    inheret from this class and implement the methods defined. Only Az-Alt
    mounts are currently supported.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_axis_names(self):
        """Get axis names

        Returns:
            A list of strings giving abbreviated names of each axis. These must
            be the keys of the dicts used by the get_position,
            get_aligned_slew_dir, remove_backlash, and get_max_slew_rates
            methods and the names accepted for the axis argument of the slew
            method.
        """

    @abc.abstractmethod
    def get_position(self, max_cache_age=0.0):
        """Gets the current position of the mount.

        Args:
            max_cache_age: If the position has been read from the mount less
                than this many seconds ago, the function may return a cached
                position value in lieu of reading the position from the mount.
                In cases where reading from the mount is relatively slow this
                may allow the function to return much more quickly. The default
                value is set to 0 seconds, in which case the function will
                never return a cached value.

        Returns:
            A dict with keys for each axis where the values are the positions
            in degrees.
        """
        pass

    @abc.abstractmethod
    def slew(self, axis, rate):
        """Command the mount to slew on one axis.

        Commands the mount to slew at a paritcular rate in one axis. A mount may enforce limits
        on the slew rate such as a maximum rate limit, acceleration limit, or max change in
        rate since the last command. Enforcement of such limits may result in the mount moving at
        a rate that is different than requested. The return values indicate whether a limit was
        exceeded and what slew rate was actually achieved.

        Args:
            axis: A string indicating the axis.
            rate: A float giving the requested slew rate in degrees per second. The sign of the
                value indicates the direction of the slew.

        Returns:
            A two-element tuple where the first element is a float giving the actual slew rate
                achieved by the mount. The actual rate could differ from the requested rate due
                to quantization or due to enforcement of slew rate, acceleration, or axis position
                limits. The second element is a boolean that is set to True if any of the limits
                enforced by the mount were exceeded by the requested slew rate.
        """
        pass


class LoopFilter(object):
    """Proportional plus integral (PI) loop filter.

    This class implements a standard proportional plus integral loop filter.
    The proportional and integral coefficients are computed on the fly in order
    to support a dynamic loop period.

    Attributes:
        bandwidth: Loop bandwidth in Hz.
        damping_factor: Loop damping factor.
        max_update_period: Maximum tolerated loop update period in seconds.
        rate_limit: Maximum allowed slew rate in degrees per second.
        accel_limit: Maximum allowed slew acceleration in degrees per second
            squared.
        step_limit: Maximum allowed change in slew rate per update in degrees per second.
        int: Integrator value.
        last_iteration_time: Unix time of last call to update().
        last_rate: Output value returned on last call to update().
    """
    def __init__(
            self,
            bandwidth,
            damping_factor,
            max_update_period=0.1
        ):
        """Inits a Loop Filter object.

        Args:
            bandwidth: Loop bandwidth in Hz.
            damping_factor: Loop damping factor.
            rate_limit: Slew rate limit in degrees per second. Set to None to remove limit.
            accel_limit: Slew acceleration limit in degrees per second squared. Set to None to
                remove limit.
            step_limit: Maximum change in slew rate allowed per update in degrees per second. Set
                to None to remove limit.
            max_update_period: Maximum tolerated loop update period in seconds.
        """
        self.bandwidth = bandwidth
        self.damping_factor = damping_factor
        self.max_update_period = max_update_period
        self.int = 0.0
        self.last_iteration_time = None
        self.last_rate = 0.0

    def clamp_integrator(self, rate):
        """Clamps the integrator magnitude to not exceed a particular rate.

        This function should be called when the slew rate given by the return value of the update()
        method trips a slew rate or acceleration limit in the mount. Otherwise the integrator can
        grow in an unbounded fashion and the system will not respond appropriately.

        Args:
            rate: The actual rate achieved by the mount after the most recent call to update(), in
                degrees per second.
        """
        self.int = clamp(self.int, abs(rate))

    def update(self, error):
        """Update the loop filter using new error signal input.

        Updates the loop filter using new error signal information. The loop
        filter proportional and integral coefficients are calculated on each
        call based on the time elapsed since the previous call. This allows
        the loop response to remain consistent even if the loop period is
        changing dynamically or can't be predicted in advance.

        If this method was last called more than max_update_period seconds ago
        a warning will be printed and the stored integrator value will be
        returned. The error signal will be ignored. This is meant to protect
        against edge cases where long periods between calls to update() could
        cause huge disturbances to the loop behavior.

        Args:
            error: The error in phase units (typically degrees).

        Returns:
            The slew rate to be applied in the same units as the error signal
                (typically degrees).
        """

        # can't measure loop period on first update
        if self.last_iteration_time is None:
            self.last_iteration_time = time.time()
            return 0.0

        update_period = time.time() - self.last_iteration_time
        self.last_iteration_time = time.time()
        if update_period > self.max_update_period:
            print('Warning: loop filter update period was {:.4f} s, limit is {:.4f} s.'.format(
                update_period,
                self.max_update_period,
            ))
            return self.int

        # compute loop filter gains based on loop period
        bt = self.bandwidth * update_period
        k0 = update_period
        denom = self.damping_factor + 1.0 / (4.0 * self.damping_factor)
        prop_gain = 4.0 * self.damping_factor / denom * bt / k0
        int_gain = 4.0 / denom**2.0 * bt**2.0 / k0

        # proportional term
        prop = prop_gain * -error

        # integral term
        self.int = self.int + int_gain * -error

        # New candidate slew rate is the sum of P and I terms. Note that the mount may enforce
        # rate or acceleration limits such that the actual rate achieved is different from this
        # rate. When this is the case, the control loop should ensure that the loop filter
        # integrator value is also saturated to prevent it from growing in an unbounded manner.
        return prop + self.int


class Tracker(TelemSource):
    """Main tracking loop class.

    This class is the core of the track package. A tracking loop or control
    loop is a system that uses feedback for control. In this case, the thing
    under control is a telescope mount. Slew commands are sent to the mount
    at regular intervals to keep it pointed in the direction of some object.
    In order to know which direction the mount should slew, the tracking loop
    needs feedback which comes in the form of an error signal. The error signal
    is a measure of the difference between where the telescope is pointed now
    compared to where the object is. The control loop tries to drive the error
    signal magnitude to zero. The final component in the loop is a loop filter
    ("loop controller" might be a better name, but "loop filter" is the name
    everyone uses). The loop filter controls the response characteristics of
    the system including the bandwidth (responsiveness to changes in input)
    and the damping factor.

    Attributes:
        loop_filter: A dict with keys for each axis where the values are
            LoopFilter objects. Each axis has its own independent loop filter.
        mount: An object of type TelescopeMount. This represents the interface
            to the mount.
        error_source: An object of type ErrorSource. The error can be computed
            in many ways so an abstract class with a generic interface is used.
        error: Cached error value returned by the error_source object's
            compute_error() method. This is cached so that callback methods
            can make use of it if needed. A dict with keys for each axis.
        slew_rate: Cached slew rates from the most recent loop filter output.
            Cached to make it available to callbacks. A dict with keys for each
            axis.
        num_iterations: A running count of the number of iterations.
        callback: A callback function. The callback will be called once at the
            end of every control loop iteration. Set to None if no callback is
            registered.
        stop: Boolean value. The control loop checks this on every iteration
            and stops if the value is True.
    """

    def __init__(self, mount, error_source, loop_bandwidth, damping_factor):
        """Inits a Tracker object.

        Initializes a Tracker object by constructing loop filters and
        initializing state information.

        Args:
            mount: Object of type TelescopeMount. Must use the same set of axes
                as error_source.
            error_source: Object of type ErrorSource. Must use the same set of
                axes as mount.
            loop_bandwidth: The loop bandwidth in Hz.
            damping_factor: The damping factor. Keep in mind that the motors in
                the mount will not respond instantaneously to slew commands,
                therefore the damping factor may need to be higher than an
                ideal system would suggest. Common values like sqrt(2)/2 may be
                too small to prevent oscillations.
        """
        if set(mount.get_axis_names()) != set(error_source.get_axis_names()):
            raise ValueError('error_source and mount must use same set of axes')
        self.loop_filter = {}
        self.loop_filter_output = {}
        self.error = {}
        self.slew_rate = {}
        for axis in mount.get_axis_names():
            self.loop_filter[axis] = LoopFilter(
                bandwidth=loop_bandwidth,
                damping_factor=damping_factor,
            )
            self.loop_filter_output[axis] = 0.0
            self.error[axis] = None
            self.slew_rate[axis] = 0.0
        self.mount = mount
        self.error_source = error_source
        self.num_iterations = 0
        self.callback = None
        self.stop = False
        self.stop_on_timer = False
        self.max_run_time = 0.0
        self.stop_when_converged = False
        self.converge_max_error_mag = 50.0 / 3600.0
        self.converge_min_iterations = 50
        self.converge_error_state = None

    def register_callback(self, callback):
        """Register a callback function.

        Registers a callback function to be called near the end of each loop
        iteration. The callback function is called with no arguments.

        Args:
            callback: The function to call. None to un-register.
        """
        self.callback = callback

    def run(self, axes=None):
        """Run the control loop.

        Call this method to start the control loop. This function is blocking
        and will not return until an error occurs or until the stop attribute
        has been set to True. Immediately after invocation this function will
        set the stop attribute to False.

        Args:
            axes: A list of strings indicating which axes should be under
                active tracking control. The slew rate on any axis not included
                in the list will not be commanded by the control loop. If an
                empty list is passed the function returns immediately. If None
                is passed all axes will be active.
        """
        self.stop = False
        low_error_iterations = 0
        start_time = time.time()

        if axes is None:
            axes = self.mount.get_axis_names()

        if len(axes) == 0:
            return 'no axes selected'

        while True:

            if self.stop:
                return 'stop flag set'

            if self.stop_when_converged and low_error_iterations >= self.converge_min_iterations:
                return 'converged'

            if self.stop_on_timer and time.time() - start_time > self.max_run_time:
                return 'timer expired'

            # compute error
            try:
                self.error = self.error_source.compute_error()
            except ErrorSource.NoSignalException:
                for axis in self.mount.get_axis_names():
                    self.error[axis] = None
                self.finish_control_cycle()
                continue

            if self.error['mag'] > self.converge_max_error_mag:
                low_error_iterations = 0
            else:
                if self.converge_error_state is None:
                    low_error_iterations += 1
                elif self.error_source.state == self.converge_error_state:
                    low_error_iterations +=1
                else:
                    low_error_iterations = 0

            # update loop filters
            for axis in axes:
                self.loop_filter_output[axis] = self.loop_filter[axis].update(self.error[axis])

            # set mount slew rates
            for axis in axes:
                (
                    self.slew_rate[axis],
                    limit_exceeded
                ) = self.mount.slew(axis, self.loop_filter_output[axis])
                if limit_exceeded:
                    self.loop_filter[axis].clamp_integrator(self.slew_rate[axis])

            self.finish_control_cycle()

    def finish_control_cycle(self):
        """Final tasks to perform at the end of each control cycle."""
        if self.callback is not None:
            self.callback(self)
        self.num_iterations += 1

    def get_telem_channels(self):
        chans = {}
        chans['num_iterations'] = self.num_iterations
        for axis in self.mount.get_axis_names():
            chans['rate_' + axis] = self.slew_rate[axis]
            chans['error_' + axis] = self.error[axis]
            chans['loop_filt_int_' + axis] = self.loop_filter[axis].int
            chans['loop_filt_out_' + axis] = self.loop_filter_output[axis]
        return chans
