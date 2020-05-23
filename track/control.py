"""tracking loop for telescope control.

Track provides the classes required to point a telescope with software using a feedback control
loop.
"""

import time
import threading
from collections import deque
from typing import NamedTuple, Tuple
import numpy as np
from scipy.optimize import minimize
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle, Longitude
from astropy.time import Time, TimeDelta
from track.errorsources import ErrorSource, PointingError
from track.model import MountModel
from track.mounts import TelescopeMount, MeridianSide, MountEncoderPositions
from track.targets import Target
from track.telem import TelemSource


def ideal_pi_gains(bandwidth, damping_factor):
    """Compute the gains for an ideal linear control system with a P+I controller.

    Note that the formulas used here are approximations that apply when the bandwidth is much
    smaller than the control loop rate.

    Reference: Equation (C.59) in Michael Rice, "Digital Communications: A Discrete-Time Approach,"
    Pearson Prentice Hall, Upper Saddle River, New Jersey, 2008.

    Args:
        bandwidth: Bandwidth of the system, typically in Hz.
        damping_factor: Damping factor (unitless).

    Returns:
        A tuple with proportional and integral gain terms. The proportional term is unitless. The
        integral gain is normalized to the loop period, i.e., it is I/T where T is the loop period,
        since this meant to be used in a discrete-time control system.
    """
    denom = damping_factor + 1.0 / (4.0 * damping_factor)
    prop_gain = 4.0 * damping_factor / denom * bandwidth
    int_gain = 4.0 / denom**2.0 * bandwidth**2
    return prop_gain, int_gain


def proportional_from_integral_and_damping(integral_gain, damping_factor):
    """Derive the proportional gain for a PI controller given the integral gain and damping factor.

    In some cases rather than starting with a bandwidth it is more practical to start with an
    integral gain, since for a second-order control system the steady-state error in response to a
    quadratic input (acceleration) is proportional to the integral gain. In the case of this
    telescope control system, the steady state error is equal to twice the target acceleration in
    degrees per second squared divided by the integral gain I/T (normalized to the loop period T).

    Args:
        integral_gain: The integral gain (I) normalized to the loop period (T), i.e., I/T.
        damping_factor: The damping factor (unitless).

    Returns:
        The proportional gain (unitless).
    """
    return 2*damping_factor*np.sqrt(integral_gain)


class MovingAverageFilter:
    """Moving average filter supporting variable sample rates.

    This is a moving average filter that maintains a consistent depth even when the sample period
    is not constant. The samples in the delay line that have a cumulative time less than or equal
    to the maximum allowed depth are averaged with equal weighting.
    """

    def __init__(self, max_depth: float = 0.2):
        """Construct a MovingAverage Filter.

        Args:
            max_depth: The maximum depth of the moving average filter in seconds. The filter will
                be applied over the last N samples where the total time duration of this set of
                samples is less than or equal to max_depth seconds.
        """
        self.delay_line: deque = deque()
        self.sample_periods: deque = deque()
        self.max_depth = max_depth

    def reset(self) -> None:
        """Reset filter state"""
        self.delay_line.clear()
        self.sample_periods.clear()

    def advance(self, value: float, sample_period: float) -> float:
        """Add a new sample to the filter and compute a new filter output.

        Args:
            value: New sample value to add to the filter.
            sample_period: The sample period of this sample in seconds. If this is greater than
                max_depth it can't be used and the return value will be 0.

        Returns:
            The new output of the filter.
        """
        self.delay_line.appendleft(value)
        self.sample_periods.appendleft(sample_period)

        if np.sum(self.sample_periods) > self.max_depth:
            # trim away samples that are too far in the past (which may include the new sample)
            discard_start_index = np.argmax(np.cumsum(self.sample_periods) > self.max_depth)
            for _ in range(len(self.delay_line) - discard_start_index):
                self.delay_line.pop()
                self.sample_periods.pop()

        return np.mean(self.delay_line) if len(self.delay_line) > 0 else 0.0


class PIDController:
    """PID loop controller

    This class implements a standard PID controller. The exact coefficients are adapted on the fly
    to support a dynamic loop period.

    Attributes:
        max_update_period: Maximum tolerated loop update period in seconds.
        integrator: Integrator value.
        last_iteration_time: Unix time of last call to update().
    """

    class PIDGains(NamedTuple):
        """Gains for PIDController

        Attributes:
            proportional: Proportional term gain.
            integral: Integral term gain.
            derivative: Derivative term gain.
            derivative_filter_depth: Max depth of the derivative moving average filter in seconds.
        """
        proportional: float = proportional_from_integral_and_damping(40.0, np.sqrt(2)/2)
        integral: float = 40.0  # allows steady state error of 0.01 deg for accel of 0.2 deg/s/s
        derivative: float = 0.0
        derivative_filter_depth: float = 0.1


    def __init__(
            self,
            gains: PIDGains = PIDGains(),
            max_update_period: float = 0.1
        ):
        """Inits a Loop Filter object.

        Args:
            gains: Set of PID gains.
            max_update_period: Maximum allowed loop update period in seconds.
        """
        self.gains = gains
        self.derivative_filter = MovingAverageFilter(gains.derivative_filter_depth)
        self.max_update_period = max_update_period
        self.integrator = 0.0
        self.error_prev = None
        self.last_iteration_time = None
        self.reset()


    def reset(self):
        """Reset to initial state."""
        self.integrator = 0.0
        self.error_prev = None
        self.derivative_filter.reset()
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
        self.integrator = np.clip(self.integrator, -abs(rate), abs(rate))


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


    def update(self, error: Angle) -> float:
        """Update the controller using new error measurement.

        Updates the controller using new error signal information. The loop filter coefficients are
        adjusted on each call based on the time elapsed since the previous call. This allows the
        system response to remain consistent even if the loop period is changing dynamically or
        can't be predicted in advance.

        If this method was last called more than max_update_period seconds ago a warning will be
        printed and the stored integrator value will be returned without any further calculation or
        change to the integrator's stored value. This is meant to protect against edge cases where
        long periods between calls to update() could cause huge disturbances to the loop behavior.

        Args:
            error: The mount axis error value.

        Returns:
            The slew rate to be applied to the mount axis. Note that the mount may enforce slew
            rate or acceleration limits such that the actual rate achieved is different from this
            rate. When this is the case, the control loop should ensure that this controller's
            integrator value is also saturated using the `clamp_integrator()` method to prevent it
            from growing in an unbounded manner (a phenomenon known as "integrator windup").
        """

        update_period = self._compute_update_period()

        # can't reliably update integrator or compute derivative term if update period is unknown
        if update_period is None:
            self.error_prev = error
            return self.gains.proportional * error + self.integrator

        # proportional term
        prop_term = self.gains.proportional * error

        # integrator update
        if update_period > self.max_update_period:
            print(f'Warning: {1e3 * update_period:.0f} ms since last iteration, '
                  f'limit is {1e3 * self.max_update_period:.0f} ms. Integrator not updated.')
        else:
            self.integrator += self.gains.integral * error * update_period

        # derivative term
        if self.error_prev is not None:
            # apply low-pass filter to derivative to reduce high-frequency noise emphasis
            diff = (error - self.error_prev) / update_period
            derivative_filter_output = self.derivative_filter.advance(diff, update_period)
            derivative_term = self.gains.derivative * derivative_filter_output
        else:
            derivative_term = 0.0
        self.error_prev = error

        return prop_term + self.integrator + derivative_term


class SlewRateCommand(NamedTuple):
    """Slew rate command to be sent at a specific time in the future.

    This is the return value of `ModelPredictiveController.update()`.

    Attributes:
        time_to_send: Time at which the slew rate commands should be sent to the mount.
        rates: Dictionary of slew rates in degrees per second where keys are the axis indices (0 or
            1).
    """
    time_to_send: Time
    rates: dict


class ModelPredictiveController:
    """A model predictive controller (MPC).

    Model predictive control uses a model of the mount dynamics to optimize slew rate commands to
    minimize predicted future pointing error. This approach is a bit computationally intensive but
    it is able to achieve excellent stability and low pointing error even when significant
    acceleration is required. An additional benefit is that stability is less dependent on the
    period of the control loop, so the stability of MPC with a somewhat slower control loop period
    may still be more stable than when using a PID controller with a somewhat faster cycle.
    """

    def __init__(
            self,
            target: Target,
            mount: TelescopeMount,
            mount_model: MountModel,
            meridian_side: MeridianSide,
            prediction_horizon: float,
            control_cycle_period: float,
        ):
        """Construct an instance of ModelPredictiveController

        Args:
            target: The target being tracked.
            mount: The mount under control. This object also provides a model of the mount used to
                predict future mount dynamics in response to control inputs.
            mount_model: Used to convert between coordinate systems.
            prediction_horizon: The controller will look ahead this many seconds into the future
                when optimizing control output against predicted future mount and target dynamics.
            control_cycle_period: The expected typical period of the control loop in seconds.
        """
        self.target = target
        self.mount = mount
        self.axes = list(mount.AxisName)
        self.mount_model = mount_model
        self.meridian_side = meridian_side
        self.prediction_horizon = TimeDelta(prediction_horizon, format='sec')
        self.control_cycle_period = TimeDelta(control_cycle_period, format='sec')
        self.slew_rate_command_prev = SlewRateCommand(
            time_to_send=Time.now(),
            rates=dict.fromkeys(self.axes, 0.0)
        )
        self._init_prediction_arrays(Time.now() + 2*self.control_cycle_period)

    def _get_target_position(self, time_future: Time) -> MountEncoderPositions:
        """Get target position for a specific time and convert to mount encoder positions"""
        target_topocentric = self.target.get_position(time_future)
        return self.mount_model.topocentric_to_encoders(target_topocentric, self.meridian_side)

    def _init_prediction_arrays(self, time_start: Time) -> None:
        """Initialize arrays of predicted target positions and slew rates out to prediction horizon

        Args:
            time_start: The first target position in the arrays will correspond to this absolute
                time. The arrays will be populated with predictions that start at this time and end
                at this time plus the prediction horizon.
        """

        num_items = int(self.prediction_horizon / self.control_cycle_period)
        times_from_start = TimeDelta(self.control_cycle_period*np.arange(num_items), format='sec')

        self.target_times = time_start + times_from_start

        self.positions_target = dict([(axis, np.zeros(num_items)) for axis in self.axes])
        for idx, target_time in enumerate(self.target_times):
            position_target = self._get_target_position(target_time)
            for axis in self.axes:
                self.positions_target[axis][idx] = position_target[axis].deg

        self.slew_rates_predicted = dict([(axis, np.zeros(num_items)) for axis in self.axes])

    def _advance_prediction_arrays(self) -> Time:
        """Advance arrays of predicted target positions and slew rates by one control cycle.

        Returns:
            The time corresponding to the oldest target position that was stored in the arrays
            prior to this method being called; in other words, the time corresponding to the target
            position removed during this call.
        """
        time_oldest_target = self.target_times[0]

        # Shift all the arrays forward by one cycle period
        self.target_times = Time(np.roll(self.target_times.value, -1))
        for axis in self.axes:
            self.positions_target[axis] = np.roll(self.positions_target[axis], -1)
            self.slew_rates_predicted[axis] = np.roll(self.slew_rates_predicted[axis], -1)

        # Update the last element of each array to a new value near the prediction horizon
        time_next = self.target_times[-2] + self.control_cycle_period
        self.target_times[-1] = time_next
        target_position = self._get_target_position(time_next)
        for axis in self.axes:
            self.positions_target[axis][-1] = target_position[axis].deg
            self.slew_rates_predicted[axis][-1] = self.slew_rates_predicted[axis][-2]

        return time_oldest_target

    def update(self) -> SlewRateCommand:
        """Run model predictive controller to generate optimized slew rate command to send next.

        Does the following:
        1. Updates arrays of predicted target positions and predicted slew rates out to horizon
        2. Predicts the position of the mount at the time the next slew rate command will be sent
        3. Computes the optimum slew rate commands to send next to minimize the average pointing
           error magnitude out to the prediction horizon.

        Returns:
            Slew rate commands to send. The return type includes both the slew rates and the time
            at which those commands should be sent. The caller should wait until that time to send
            the commands rather than sending them immediately.
        """

        # get current position and slew rates of the mount
        position_mount_now = self.mount.get_position()
        slew_rates_now = tuple([self.mount.get_slew_rate(axis) for axis in self.axes])
        time_now = Time.now()

        # refresh arrays of predicted target positions looking into the future
        while True:
            time_next_cmd = self._advance_prediction_arrays()
            time_next_cmd_from_now = time_next_cmd - time_now
            # Keep advancing if the time the next command should be sent is already in the past.
            # Hopefully this does not happen often, since it means that the control loop is falling
            # behind. If keeping up, time_next_cmd_from_now should be nearly equal to the control
            # cycle period.
            if time_next_cmd_from_now > 0:
                break
        times_from_next_cmd = self.target_times - time_next_cmd


        for axis in self.axes:

            # predict mount position and slew rate at the instant when the next slew rate command
            # will be sent, which is the start of the prediction window used by the optimizer
            position_mount_at_next_cmd, slew_rate_at_next_cmd = self.mount.predict(
                times_from_start=np.array([time_next_cmd_from_now.sec]),
                rate_commands=np.array([self.slew_rate_command_prev.rates[axis]]),
                position_axis_start=position_mount_now[axis].deg,
                slew_rate_start=slew_rates_now[axis]
            )
            # Numpy arrays returned by previous call to mount.predict() are length-1
            position_mount_at_next_cmd = float(position_mount_at_next_cmd)
            slew_rate_at_next_cmd = float(slew_rate_at_next_cmd)

            # run optimizer to find best slew rate command to send next
            opt_result = minimize(
                fun=self._objective,
                x0=self.slew_rates_predicted[axis],
                args=(
                    times_from_next_cmd.sec,
                    self.positions_target[axis],
                    position_mount_at_next_cmd,
                    slew_rate_at_next_cmd,
                    self.mount.no_cross_encoder_positions()[axis].deg,
                ),
                bounds=[(
                    -self.mount.max_slew_rate,
                    self.mount.max_slew_rate
                )]*len(times_from_next_cmd),
                method='SLSQP',  # Chosen by experimentation for speed and consistency
                options={'disp': False, 'ftol': 1e-10, 'maxiter': 10}
            )
            self.slew_rates_predicted[axis] = opt_result.x

        slew_rate_command = SlewRateCommand(
            time_to_send=time_next_cmd,
            rates=dict([(axis, self.slew_rates_predicted[axis][0]) for axis in self.axes])
        )
        self.slew_rate_command_prev = slew_rate_command

        return slew_rate_command

    def _objective(
            self,
            slew_rate_commands: np.ndarray,
            times_from_start: np.ndarray,
            positions_target: np.ndarray,
            position_axis_start: float,
            slew_rate_start: float,
            no_cross_position: float,
        ):
        """The objective (cost) function that the optimizer attempts to minimize.

        This predicts the mean magnitude of the pointing error over a future time window in
        response to a provided set of future slew rate commands.
        """

        # predict mount axis position in the future assuming this set of slew rate commands
        positions_mount, _ = self.mount.predict(
            times_from_start,
            slew_rate_commands,
            position_axis_start,
            slew_rate_start
        )

        pointing_errors = ErrorSource._smallest_allowed_error(
            positions_mount,
            positions_target,
            no_cross_position
        )

        # return mean error magnitude
        return np.mean(np.abs(pointing_errors))


class Tracker(TelemSource):
    """Main tracking loop class.

    This class is the core of the track package. It forms a closed-loop control system. The thing
    under control is a telescope mount. Slew commands are sent to the mount at regular intervals to
    keep it pointed in the direction of some target object. In order to know which direction and
    speed the mount should slew, the system needs feedback which comes in the form of an error
    signal. The error signal is a measure of the difference between where the target object is and
    where the telescope is pointed now. The control loop tries to drive the error signal magnitude
    to zero. The final component in the loop is a PID controller which has adjustable gains that
    are tuned to optimize the response characteristics of the system such as steady-state error,
    tracking noise, and how long it takes for the mount to reach small pointing error when slewing
    to a new target.

    Since there are two axes in the telescope mount this class really contains two control systems,
    one for each mount axis. For each axis, the error terms, controller, and mount actuation are
    in terms of the encoder and slew rate associated with that mount axis. In some cases it is
    most natural to think of pointing error and positions in terms of celestial coordiante systems
    such as topocentric (azimuth and altitude) or equatorial (right ascension and declination), but
    these coordinates must be transformed to axis encoder positions for use in this control system.

    Attributes:
        controller: A dict with keys for each axis where the values are LoopController objects.
            Each mount axis has its own independent controller.
        mount: An object of type TelescopeMount. This represents the interface to the mount.
        error_source: An object of type ErrorSource. The error can be computed in many ways so an
            abstract class with a generic interface is used.
        error: Cached error value returned by the error_source object's compute_error() method.
            This is cached so that callback methods can make use of it if needed. A dict with keys
            for each axis.
        slew_rate: Cached slew rates from the most recent loop filter output. Cached to make it
            available to callbacks. A dict with keys for each axis.
        num_iterations: A running count of the number of iterations.
        callback: A callback function. The callback will be called during every control loop
            iteration. None if no callback is registered.
        stop: Boolean value. The control loop checks this on every iteration and stops if the
            value is True.
    """


    def __init__(
            self,
            mount: TelescopeMount,
            mount_model: MountModel,
            meridian_side: MeridianSide,
            target: Target,
            control_loop_period: float = 0.1,
        ):
        """Constructs a Tracker object.

        Args:
            mount: Provides interface to send slew rate commands to the mount.
            mount_model: Alignment model for converting to/from mount encoder positions.
            meridian_side: Selects which side of mount meridian to point in.
            target: The target to track.
            control_loop_period: Target control loop period in seconds.
        """
        self.axes = list(mount.AxisName)
        self.controller = ModelPredictiveController(
            target=target,
            mount=mount,
            mount_model=mount_model,
            meridian_side=meridian_side,
            prediction_horizon=1.0,
            control_cycle_period=control_loop_period,
        )
        self.control_loop_period = control_loop_period
        self.controller_outputs = dict.fromkeys(self.axes, 0.0)
        self.error = PointingError(None, None, None)
        self.slew_rate = dict.fromkeys(self.axes, 0.0)
        self.mount = mount
        self.num_iterations = 0
        self.callback = None
        self.stop = False
        self.stop_on_timer = False
        self.max_run_time = 0.0
        self.stop_when_converged = False
        self.converge_max_error_mag = Angle(50.0 / 3600.0 * u.deg)
        self.converge_min_iterations = 50
        self.converge_error_state = None
        self._telem_mutex = threading.Lock()
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


    def run(self) -> str:
        """Run the control loop.

        Starts the control loop. This function is blocking and will not return until an error
        occurs or until one of the various stopping conditions is satisfied.

        Returns:
            A string giving the reason the control loop stopped.
        """
        self.stop = False
        low_error_iterations = 0
        start_time = time.time()

        while True:

            if self.stop:
                return 'stop flag set'

            if self.stop_when_converged and low_error_iterations >= self.converge_min_iterations:
                return 'converged'

            if self.stop_on_timer and time.time() - start_time > self.max_run_time:
                return 'timer expired'

            if self.callback is not None:
                if self.callback(self):
                    self._finish_control_cycle()
                    continue

            if not any(self.error):
                self._finish_control_cycle()
                continue

            if self.error.magnitude > self.converge_max_error_mag:
                low_error_iterations = 0
            else:
                if self.converge_error_state is None:
                    low_error_iterations += 1
                elif self.error_source.state == self.converge_error_state:
                    low_error_iterations += 1
                else:
                    low_error_iterations = 0

            # update loop controllers
            ctrl_out = self.controller.update()
            for idx, axis in enumerate(self.axes):
                self.controller_outputs[axis] = ctrl_out[idx]

            # set mount slew rates
            for axis in self.axes:
                (
                    self.slew_rate[axis],
                    limit_exceeded
                ) = self.mount.slew(axis, self.controller_outputs[axis])

            self._finish_control_cycle()


    def _finish_control_cycle(self):
        """Final tasks to perform at the end of each control cycle."""
        self._set_telem_channels()
        self.num_iterations += 1


    def _set_telem_channels(self):
        self._telem_mutex.acquire()
        self._telem_chans = {}
        self._telem_chans['num_iterations'] = self.num_iterations
        for axis in self.axes:
            self._telem_chans[f'rate_{axis}'] = self.slew_rate[axis]
            try:
                self._telem_chans[f'error_{axis}'] = self.error[axis].deg
            except AttributeError:
                pass
            self._telem_chans[f'controller_out_{axis}'] = self.controller_outputs[axis]
        self._telem_mutex.release()


    def get_telem_channels(self):
        self._telem_mutex.acquire()
        chans = self._telem_chans.copy()
        self._telem_mutex.release()
        return chans
