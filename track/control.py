"""tracking loop for telescope control.

Track provides the classes required to point a telescope with software using a feedback control
loop.
"""

import time
import threading
from enum import Flag, auto
from typing import Callable, NamedTuple, Tuple, Optional, Union
import numpy as np
from scipy.optimize import minimize
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle, UnitSphericalRepresentation
from astropy.time import Time, TimeDelta
from track.model import MountModel
from track.mounts import TelescopeMount, MountEncoderPositions
from track.targets import Target
from track.telem import TelemSource


def separation(sc1: SkyCoord, sc2: SkyCoord) -> Angle:
    """Calculate the on-sky separation angle between two coordinates.

    This is equivalent to `SkyCoord.separation()` but is much faster because it uses the haversine
    formula rather than the iterative Vincenty formula. We don't need to handle edge cases like
    antipodal points on the sphere but we do need fast execution. This approach was profiled as
    ~70 times faster than `SkyCoord.separation()`.

    Formula reference: https://en.wikipedia.org/wiki/Great-circle_distance

    Args:
        sc1: One of the coordinates.
        sc2: The other coordinate.

    Returns:
        The separation between sc1 and sc2.
    """
    us1 = sc1.represent_as(UnitSphericalRepresentation)
    us2 = sc2.represent_as(UnitSphericalRepresentation)
    lat_diff = us1.lat.rad - us2.lat.rad
    lon_diff = us1.lon.rad - us2.lon.rad
    return Angle(
        2 * np.arcsin(np.sqrt(
            np.sin(lat_diff / 2)**2
            + np.cos(us1.lat.rad)*np.cos(us2.lat.rad)*np.sin(lon_diff / 2)**2
        )) * u.rad
    )


def smallest_allowed_error(
        mount_enc_position: Union[float, np.ndarray],
        target_enc_position: Union[float, np.ndarray],
        no_cross_position: Optional[float] = None,
    ) -> np.ndarray:
    """Compute error term for a single axis taking into account no-cross positions

    The shortest path on an axis is just the difference between the target and mount encoder
    positions wrapped to [-180, +180] degrees. However the shortest path is not always allowed
    because it may result in cord wrap or send the mount moving towards its built-in axis
    limits. This function contains the logic needed to account for this, resulting in error
    terms that are either the shortest distance if allowed or the longer distance if
    constraints prevent taking the short path.

    Note that Astropy objects are not used for performance reasons. Otherwise, the arguments
    would be Longitude objects and the return type would be an Angle.

    Args:
        mount_enc_position: The mount encoder position(s) in degrees. Scalar or array.
        target_enc_position: The target encoder position(s) in degrees. Scalar or array.
        no_cross_position: Encoder position in degrees that this axis is not allowed to cross
            or None if no such position exists.

    Returns:
        The error term(s) for this axis in degrees.
    """

    # shortest path to target
    prelim_error = (target_enc_position - mount_enc_position + 180) % 360 - 180

    if no_cross_position is None:
        return prelim_error

    # error term as if the no-cross point were the target position
    no_cross_error = (no_cross_position - mount_enc_position + 180) % 360 - 180

    # actual target is closer than the no-cross point so can't possibly be crossing it
    target_closer_indices = np.abs(prelim_error) <= np.abs(no_cross_error)

    # actual target is in opposite direction from the no-cross point
    target_opposite_indices = np.sign(prelim_error) != np.sign(no_cross_error)

    final_error = np.where(
        np.logical_or(target_closer_indices, target_opposite_indices),
        prelim_error,
        # switch direction since prelim_error would have crossed the no-cross point
        prelim_error - 360*np.sign(prelim_error)
    )

    return final_error


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


class MountState(NamedTuple):
    """Positions and slew rates of mount as queried at a particular time.

    Attributes:
        time_queried: The time when the positions and slew rates were queried. Since these can't
            necessarily be queried at the same instant this timestamp may have an error that is
            different for each of the measurements.
        position: The positions of the mount axes.
        rates: The slew rates of the mount axes in degrees per second.
    """
    time_queried: Time
    position: MountEncoderPositions
    rates: Tuple


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
            prediction_horizon: float,
            control_cycle_period: float,
        ):
        """Construct an instance of ModelPredictiveController

        Args:
            target: The target being tracked.
            mount: The mount under control. This object also provides a model of the mount used to
                predict future mount dynamics in response to control inputs.
            prediction_horizon: The controller will look ahead this many seconds into the future
                when optimizing control output against predicted future mount and target dynamics.
            control_cycle_period: The expected typical period of the control loop in seconds.
        """
        self.mount = mount
        self.axes = list(mount.AxisName)
        self.prediction_horizon = TimeDelta(prediction_horizon, format='sec')
        self.control_cycle_period = TimeDelta(control_cycle_period, format='sec')
        self.slew_rate_command_prev = SlewRateCommand(
            time_to_send=Time.now(),
            rates=dict.fromkeys(self.axes, 0.0)
        )

        # These attributes will be initialized when target property is set
        self.target_times = None
        self.positions_target = None
        self.slew_rates_predicted = None

        self.target = target  # setting this property triggers array initialization

    @property
    def target(self) -> Target:
        """Get target"""
        return self._target

    @target.setter
    def target(self, new_target: Target) -> None:
        self._target = new_target
        self._init_prediction_arrays(Time.now() + 2*self.control_cycle_period)

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

        # init target position array to current mount position in case actual target position
        # is indeterminate
        position_mount = self.mount.get_position()
        self.positions_target = {
            axis: np.full(num_items, position_mount[axis].deg) for axis in self.axes
        }
        self._refresh_target_positions()

        self.slew_rates_predicted = {axis: np.zeros(num_items) for axis in self.axes}

    def _refresh_target_positions(self) -> None:
        """Refresh all of the elements of the predicted target position arrays

        Can't assume for all target types that predicted positions will remain unchanged as time
        progresses. For example, a prediction ten seconds into the future may have low confidence
        but 9 seconds later, when that time instant is only one second in the future, the estimate
        of the future position may have been greatly refined.
        """
        for idx, target_time in enumerate(self.target_times):
            try:
                position_target = self.target.get_position(target_time)
            except self.target.IndeterminatePosition:
                continue
            for axis in self.axes:
                self.positions_target[axis][idx] = position_target.enc[axis].deg

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

        # Update the last element of time and slew rate arrays
        self.target_times[-1] = self.target_times[-2] + self.control_cycle_period
        for axis in self.axes:
            self.slew_rates_predicted[axis][-1] = self.slew_rates_predicted[axis][-2]
            # This should be overwritten by `_refresh_target_positions()` but in case it is unable
            # to get a new prediction this is the next-best thing
            self.positions_target[axis][-1] = self.positions_target[axis][-2]

        # refresh predicted target positions
        self._refresh_target_positions()

        return time_oldest_target

    def update(self, mount_state: MountState) -> SlewRateCommand:
        """Run model predictive controller to generate optimized slew rate command to send next.

        Does the following:
        1. Updates arrays of predicted target positions and predicted slew rates out to horizon
        2. Predicts the position of the mount at the time the next slew rate command will be sent
        3. Computes the optimum slew rate commands to send next to minimize the average pointing
           error magnitude out to the prediction horizon.

        Args:
            mount_state: State of the mount (axis positions and slew rates). This state should be
                as fresh as possible.

        Returns:
            A set of slew rates to be commanded and the time at which those commands should
            be sent. The caller should wait until that time to send the commands rather than
            sending them immediately.
        """

        # refresh arrays of predicted target positions looking into the future
        while True:
            time_next_cmd = self._advance_prediction_arrays()
            time_next_cmd_from_now = time_next_cmd - mount_state.time_queried
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
                position_axis_start=mount_state.position[axis].deg,
                slew_rate_start=mount_state.rates[axis]
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
            rates={axis: self.slew_rates_predicted[axis][0] for axis in self.axes}
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
            no_cross_position: Optional[float] = None,
        ) -> float:
        """The objective (cost) function that the optimizer attempts to minimize.

        This predicts the mean magnitude of the pointing error over a future time window in
        response to a provided set of future slew rate commands.

        Args:
            slew_rate_commands: Array of future slew rate commands in degrees per second.
            times_from_start: Array of times measured in seconds from the start time. The start
                time (t=0) is defined as the time at which the next slew rate command will be
                issued (the first element of slew_rate_commands). Each of these times maps 1:1 to
                the elements of the positions_target array. The future position of the mount will
                be predicted for these times in this method.
            positions_target: Array of predicted target positions for this mount axis at the times
                indicated in the times_from_start array. The predicted position of the mount axis
                will be compared to these.
            position_axis_start: The position of the mount axis to assume at the start time.
            slew_rate_start: The slew rate of the mount axis to assume at the start time.
            no_cross_position: A mount axis position that should not be crossed.

        Returns:
            The predicted mean error magnitude over the prediction window.
        """

        # predict mount axis position in the future assuming this set of slew rate commands
        positions_mount, _ = self.mount.predict(
            times_from_start,
            slew_rate_commands,
            position_axis_start,
            slew_rate_start
        )

        pointing_errors = smallest_allowed_error(
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
    keep it pointed in the direction of some target object. A model predictive controller
    determines what rate commands to send to the mount next to minimize the predicted pointing
    error over a future time horizon (see `ModelPredictiveController` for more details).

    Since there are two axes in a typical telescope mount this class really contains two control
    systems running in parallel, one for each mount axis.
    """

    class StoppingConditions(NamedTuple):
        """Tracker stopping conditions. This is passed as an argument to `Tracker.run()`.

        Attributes:
            timeout: run() method will return after at most this many seconds when not None. The
                TIMEOUT flag in the return value will be set when this occurs.
            error_threshold: run() method will return when the pointing error is less than this
                when not None. The CONVERGED flag in the return value will be set when this
                criterion is met.
        """
        timeout: float
        error_threshold: Angle

    class StopReason(Flag):
        """Tracker `run()` method return value indicating stop reason or reasons."""
        NONE = 0
        TIMEOUT = auto()
        CONVERGED = auto()

    def __init__(
            self,
            mount: TelescopeMount,
            mount_model: MountModel,
            target: Target,
            control_loop_period: float = 0.1,
        ):
        """Constructs a Tracker object.

        Args:
            mount: Provides interface to send slew rate commands to the mount.
            mount_model: Alignment model for converting to/from mount encoder positions.
            target: The target to track.
            control_loop_period: Target control loop period in seconds.
        """
        self.axes = list(mount.AxisName)
        self.controller = ModelPredictiveController(
            target=target,
            mount=mount,
            prediction_horizon=1.0,
            control_cycle_period=control_loop_period,
        )
        self.control_loop_period = control_loop_period
        self.mount = mount
        self.mount_model = mount_model
        self.target = target
        self.num_iterations = 0
        self.callback = None

        # these are set when `run()` is called
        self.stopping_conditions = None
        self.time_run_called = None

        self._telem_mutex = threading.Lock()
        self._telem_chans = {}

    @property
    def target(self) -> Target:
        """Get target"""
        return self._target

    @target.setter
    def target(self, new_target: Target) -> None:
        self._target = new_target
        self.controller.target = new_target

    def register_callback(self, callback: Callable[["Tracker"], bool]) -> None:
        """Register a callback function.

        Registers a callback function to be called once per control loop iteration. The callback
        function is passed a single argument which is a reference to this object (self). It must
        return a boolean where if False the remainder of that control loop cycle will execute as
        normal, or when True the rest of the control loop cycle is skipped (other than setting
        telemetry channels and incrementing the iteration counter). Thus the callback is
        effectively able to hijack the behavior of the control loop.

        Args:
            callback: The function to call. None to un-register.
        """
        self.callback = callback


    def run(self, stopping_conditions: Optional[StoppingConditions] = None) -> "Tracker.StopReason":
        """Run the control loop.

        Starts the control loop. This function is blocking and will not return until an error
        occurs or until one of the various stopping conditions is satisfied.

        Args:
            stopping_conditions: A set of stopping conditions that must be satisfied for this to
                return. If None this will never return.

        Returns:
            A set of flags indicating which stopping conditions were met that caused this method
            to return.
        """
        self.time_run_called = time.perf_counter()
        self.stopping_conditions = stopping_conditions
        time_last_start = None

        while True:

            # measure control cycle period
            time_now = time.perf_counter()
            cycle_period = time_now - time_last_start if time_last_start is not None else None
            time_last_start = time_now

            # some (but not all) targets need to grab and process data from sensors
            self.target.process_sensor_data()

            # get current position and slew rates of the mount
            mount_state = MountState(
                rates=tuple([self.mount.get_slew_rate(axis) for axis in self.axes]),
                position=self.mount.get_position(),
                time_queried=Time.now(),
            )

            if self.callback is not None:
                if self.callback(self):
                    stop_reason = self._finish_control_cycle(
                        cycle_period,
                        mount_state,
                        callback_override=True
                    )
                    if stop_reason != self.StopReason.NONE:
                        return stop_reason
                    continue

            # get next set of slew rate commands
            rate_command = self.controller.update(mount_state)

            # sleep until it's time to send the commands
            time_to_sleep = rate_command.time_to_send - Time.now()
            if time_to_sleep > 0:
                time.sleep(time_to_sleep.sec)

            rate_command_time_error = Time.now() - rate_command.time_to_send

            # set mount slew rates as close in time as possible to when the controller wanted
            for axis in self.axes:
                self.mount.slew(axis, rate_command.rates[axis])

            stop_reason = self._finish_control_cycle(
                cycle_period,
                mount_state,
                rate_command,
                rate_command_time_error
            )
            if stop_reason != self.StopReason.NONE:
                return stop_reason


    def _finish_control_cycle(
            self,
            cycle_period: Optional[float],
            mount_state: MountState,
            rate_command: Optional[SlewRateCommand] = None,
            rate_command_time_error: Optional[float] = None,
            callback_override: bool = False,
        ) -> "Tracker.StopReason":
        """Final tasks to perform at the end of each control cycle."""

        # coordinate system transformations
        position_mount_topo = self.mount_model.encoders_to_topocentric(mount_state.position)

        try:
            # get target position for the same time as mount state was queried
            position_target = self.target.get_position(mount_state.time_queried)

        except Target.IndeterminatePosition:
            stop_reason = self._check_stopping_conditions()
            self._telem_mutex.acquire()
            self._telem_chans = {}

        else:
            error_enc = {axis: float(smallest_allowed_error(
                mount_state.position[axis].deg,
                position_target.enc[axis].deg,
                self.mount.no_cross_encoder_positions()[axis].deg,
            )) for axis in self.axes}

            # on-sky separation between target and mount positions
            error_magnitude = separation(position_target.topo, position_mount_topo)

            stop_reason = self._check_stopping_conditions(error_magnitude)

            # populate dict of telemetry channels
            self._telem_mutex.acquire()
            self._telem_chans = {}
            self._telem_chans['target_az'] = position_target.topo.az.deg
            self._telem_chans['target_alt'] = position_target.topo.alt.deg
            self._telem_chans['error_mag'] = error_magnitude.deg
            for axis in self.axes:
                self._telem_chans[f'target_enc_{axis}'] = position_target.enc[axis].deg
                self._telem_chans[f'error_enc_{axis}'] = error_enc[axis]


        if cycle_period is not None:
            self._telem_chans['cycle_period'] = cycle_period
        self._telem_chans['num_iterations'] = self.num_iterations
        self._telem_chans['callback_override'] = int(callback_override)
        self._telem_chans['mount_az'] = position_mount_topo.az.deg
        self._telem_chans['mount_alt'] = position_mount_topo.alt.deg
        for axis in self.axes:
            self._telem_chans[f'rate_{axis}'] = mount_state.rates[axis]
            self._telem_chans[f'mount_enc_{axis}'] = mount_state.position[axis].deg
            if rate_command is not None:
                self._telem_chans[f'rate_command_{axis}'] = rate_command.rates[axis]
        if rate_command_time_error is not None:
            self._telem_chans['rate_command_time_error'] = rate_command_time_error.sec
        self._telem_mutex.release()

        self.num_iterations += 1
        return stop_reason


    def _check_stopping_conditions(
            self,
            error_magnitude: Optional[Angle] = None
        ) -> "Tracker.StopReason":
        """Checks if any set stopping conditions are satisfied.

        Args:
            error_magnitude: Current pointing error magnitude.

        Returns:
            An instance of StopReason with flags set to indicate which stopping conditions are
            currently met, if any.
        """
        stop_reason = self.StopReason.NONE

        if self.stopping_conditions is None:
            return stop_reason

        if self.stopping_conditions.timeout is not None:
            if time.perf_counter() - self.time_run_called > self.stopping_conditions.timeout:
                stop_reason |= self.StopReason.TIMEOUT

        if self.stopping_conditions.error_threshold is not None:
            if error_magnitude is not None:
                if error_magnitude <= self.stopping_conditions.error_threshold:
                    stop_reason |= self.StopReason.CONVERGED

        return stop_reason

    def get_telem_channels(self):
        self._telem_mutex.acquire()
        chans = self._telem_chans.copy()
        self._telem_mutex.release()
        return chans
