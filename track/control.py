"""tracking loop for telescope control.

Track provides the classes required to point a telescope with software using a feedback control
loop.
"""

import time
import threading
from track.errorsources import ErrorSource, PointingError
from track.telem import TelemSource
from track.mathutils import clamp


class LoopFilter(object):
    """Proportional plus integral (PI) loop filter.

    This class implements a standard proportional plus integral loop filter. The proportional and
    integral coefficients are computed on the fly in order to support a dynamic loop period.

    Attributes:
        bandwidth: Loop bandwidth in Hz.
        damping_factor: Loop damping factor.
        max_update_period: Maximum tolerated loop update period in seconds.
        int: Integrator value.
        last_iteration_time: Unix time of last call to update().
    """
    def __init__(
            self,
            bandwidth,
            damping_factor,
            max_update_period=0.1
        ):
        """Inits a Loop Filter object.

        Args:
            bandwidth (float): Loop bandwidth in Hz.
            damping_factor (float): Loop damping factor.
            max_update_period (float): Maximum tolerated loop update period in seconds.
        """
        self.bandwidth = bandwidth
        self.damping_factor = damping_factor
        self.max_update_period = max_update_period
        self.reset()


    def reset(self):
        """Reset to initial state."""
        self.int = 0.0
        self.last_iteration_time = None


    def clamp_integrator(self, rate):
        """Clamps the integrator magnitude to not exceed a particular rate.

        This function should be called when the slew rate given by the return value of the update()
        method trips a slew rate or acceleration limit in the mount. Otherwise the integrator can
        grow in an unbounded fashion and the system will not respond appropriately.

        Args:
            rate (float): The integrator will be clamped such that it does not exceed the magnitude
                of this value.
        """
        self.int = clamp(self.int, abs(rate))


    def _compute_update_period(self):
        """Determine the time elapsed since last call to update()

        Returns:
            float: Seconds since last call to update() or None if the time of the previous call to
                update() is not known, which will be the case before the first call to update() or
                if reset() has been called since the last call to update().
        """
        if self.last_iteration_time is None:
            self.last_iteration_time = time.perf_counter()
            return None

        now = time.perf_counter()
        update_period = now - self.last_iteration_time
        self.last_iteration_time = now
        return update_period


    def update(self, error):
        """Update the loop filter using new error signal input.

        Updates the loop filter using new error signal information. The loop filter proportional
        and integral coefficients are calculated on each call based on the time elapsed since the
        previous call. This allows the loop response to remain consistent even if the loop period
        is changing dynamically or can't be predicted in advance.

        If this method was last called more than max_update_period seconds ago a warning will be
        printed and the stored integrator value will be returned without any further calculation or
        change to the integrator's stored value. This is meant to protect against edge cases where
        long periods between calls to update() could cause huge disturbances to the loop behavior.

        Args:
            error (float): The error in phase units (typically degrees).

        Returns:
            float: The slew rate to be applied in the same units as the error signal.
        """

        update_period = self._compute_update_period()

        # can't reliably compute loop filter coefficients if period is not known
        if update_period is None:
            return self.int

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
        self.axes = list(mount.AxisName)
        self.loop_filter = dict.fromkeys(self.axes, LoopFilter(loop_bandwidth, damping_factor))
        self.loop_filter_output = dict.fromkeys(self.axes, 0.0)
        self.error = PointingError(None, None)
        self.slew_rate = dict.fromkeys(self.axes, 0.0)
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
        self.telem_mutex = threading.Lock()
        self._set_telem_channels()


    def register_callback(self, callback):
        """Register a callback function.

        Registers a callback function to be called once per control loop iteration, just after
        the error is calculated but prior to any subsequent processing. The callback function is
        passed a single argument which is a reference to this object (self). It must return a
        boolean where if False the remainder of that control loop cycle will execute as normal, or
        when True the rest of the control loop cycle is skipped (other than setting telemetry
        channels and incrementing the iteration counter). Thus the callback is effectively able to
        hijack the behavior of the control loop.

        Args:
            callback: The function to call. None to un-register.
        """
        self.callback = callback


    def run(self, axes=None):
        """Run the control loop.

        Call this method to start the control loop. This function is blocking and will not return
        until an error occurs or until the stop attribute has been set to True. Immediately after
        invocation this function will set the stop attribute to False.
        """
        self.stop = False
        low_error_iterations = 0
        start_time = time.time()

        for axis in self.axes:
            self.loop_filter[axis].reset()

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
                self.error = PointingError(None, None)

            if self.callback is not None:
                if self.callback(self):
                    self._finish_control_cycle()
                    continue

            if not any(self.error):
                self._finish_control_cycle()
                continue

            # TODO: Fix this
            # if self.error['mag'] > self.converge_max_error_mag:
            #     low_error_iterations = 0
            # else:
            #     if self.converge_error_state is None:
            #         low_error_iterations += 1
            #     elif self.error_source.state == self.converge_error_state:
            #         low_error_iterations +=1
            #     else:
            #         low_error_iterations = 0

            # update loop filters
            for axis in self.axes:
                self.loop_filter_output[axis] = self.loop_filter[axis].update(self.error[axis.value].deg)

            # set mount slew rates
            for axis in self.axes:
                (
                    self.slew_rate[axis],
                    limit_exceeded
                ) = self.mount.slew(axis, self.loop_filter_output[axis])
                if limit_exceeded:
                    self.loop_filter[axis].clamp_integrator(self.slew_rate[axis])

            self._finish_control_cycle()


    def _finish_control_cycle(self):
        """Final tasks to perform at the end of each control cycle."""
        self._set_telem_channels()
        self.num_iterations += 1


    def _set_telem_channels(self):
        self.telem_mutex.acquire()
        self.telem_chans = {}
        # TODO: Fix this
        # self.telem_chans['num_iterations'] = self.num_iterations
        # for axis in self.axes:
        #     axis_name = axis.short_name()
        #     self.telem_chans['rate_' + axis_name] = self.slew_rate[axis]
        #     self.telem_chans['error_' + axis_name] = self.error[axis.value]
        #     self.telem_chans['loop_filt_int_' + axis_name] = self.loop_filter[axis].int
        #     self.telem_chans['loop_filt_out_' + axis_name] = self.loop_filter_output[axis]
        self.telem_mutex.release()


    def get_telem_channels(self):
        self.telem_mutex.acquire()
        chans = self.telem_chans.copy()
        self.telem_mutex.release()
        return chans
