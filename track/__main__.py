#!/usr/bin/env python3

"""Automated tracking of objects.

This is the core program in this project. The software uses a control system to cause a telescope
mount to track a target based on the target's position in the sky and the mount's encoder readings.
"""
from __future__ import annotations
from functools import partial
import logging
import os
import signal
import sys
from typing import Callable
import click
import astropy.units as u
from astropy.coordinates import Angle, Longitude
from track import control, laser, logs, model, mounts, ntp, targets, telem
from track.config import ArgParser, CONFIG_PATH
from track.control import Tracker
from track.gamepad import Gamepad
from track.mounts import MeridianSide


logger = logging.getLogger(__name__)


def make_camera_separation_callback(
        pid_to_signal: int | None = None,
        separation_threshold: float | None = None,
    ) -> Callable[[Angle], None] | None:
    """Make callable that sends SIGUSR1 to another process when target is near center of camera.

    The purpose of this is to provide a synchronization mechanism between this program and the
    `align_guidescope` program, which needs to wait until a star is centered in the camera frame
    before proceeding.

    Args:
        pid_to_signal: The process id (PID) to which SIGUSR1 will be sent by the returned function.
        separation_threshold: The returned function accepts a separation angle as the single
            argument. When the argument is below this threshold in degrees, the function will send
            SIGUSR1.

    Returns:
        A function that may be passed to `targets.make_target_from_args()`, or None if both
        arguments are None.
    """
    def camera_separation_callback(
        separation: Angle,
        separation_threshold: Angle,
        pid: int,
    ) -> None:
        if separation < separation_threshold:
            os.kill(pid, signal.SIGUSR1)

    if pid_to_signal is not None:
        logger.debug(f'Will send SIGUSR1 to PID {pid_to_signal} when star is centered in camera.')
        return partial(
            camera_separation_callback,
            pid=pid_to_signal,
            separation_threshold=Angle(u.deg * separation_threshold)
        )
    return None


def main():
    """See module docstring"""

    def gamepad_callback(_unused: Tracker) -> bool:
        """Callback for gamepad control.

        Allows manual control of the slew rate via a gamepad when the 'B' button is held down,
        overriding tracking behavior. This callback is registered with the Tracker object which
        calls it on each control cycle.

        Defined inside main() so this has easy access to objects that are within that scope.

        Returns:
            True when the 'B' button is depressed which disables the normal control path. False
                otherwise, which leaves the normal control path enabled.
        """
        if game_pad.state.get('BTN_EAST', 0) == 1:
            gamepad_x, gamepad_y = game_pad.get_proportional()
            slew_rates = (
                mount.max_slew_rate * gamepad_x,
                mount.max_slew_rate * gamepad_y
            )
            for idx, axis_name in enumerate(mount.AxisName):
                mount.slew(axis_name, slew_rates[idx])
            return True
        else:
            return False


    parser = ArgParser(additional_config_files=[os.path.join(CONFIG_PATH, 'track.cfg')])
    targets.add_program_arguments(parser)
    laser.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    mounts.add_program_arguments(parser, meridian_side_required=True)
    ntp.add_program_arguments(parser)
    telem.add_program_arguments(parser)
    control.add_program_arguments(parser)
    signal_group = parser.add_argument_group(
        title='Signal Options',
        description='Options that pertain to sending synchronization signals to other processes',
    )
    signal_group.add_argument(
        '--pid-to-signal',
        help='send SIGUSR1 to the process with this PID when target is within signal-angle of '
            'camera frame center',
        type=int,
    )
    signal_group.add_argument(
        '--signal-angle',
        help='SIGUSR1 sent when target to frame center separation is below this angle in degrees',
        type=float,
    )
    args = parser.parse_args()

    if args.pid_to_signal is not None and args.signal_angle is None:
        # pylint: disable=not-callable
        parser.error('--pid-to-signal requires --signal-angle')

    logs.setup_logging_from_args(args, __package__)

    # Set priority of this thread to realtime. Do this before constructing objects since priority
    # is inherited and some critical threads are created by libraries we have no direct control
    # over.
    os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(11))

    # Check if system clock is synchronized to GPS
    if args.check_time_sync:
        try:
            ntp.check_ntp_status()
        except ntp.NTPCheckFailure:
            logger.exception('NTP check failed.')
            if not click.confirm('Continue anyway?', default=True):
                sys.exit(2)

    # Load a MountModel object
    try:
        mount_model = model.load_stored_model()
    except model.StaleParametersException:
        if click.confirm('Stored alignment parameters are stale. Use anyway?', default=True):
            mount_model = model.load_stored_model(max_age=None)
        elif args.target_type == 'camera':
            if click.confirm('Use a default set of alignment parameters instead?', default=True):
                guide_cam_orientation = click.prompt(
                    'Enter guide camera orientation in degrees, clockwise positive', type=float)
                mount_model = model.load_default_model(
                    guide_cam_orientation=Longitude(guide_cam_orientation*u.deg))

    if 'mount_model' not in locals():
        logger.error('No model could be loaded. To refresh stored model run align program.')
        sys.exit(1)

    logger.info(f'Using mount model with the following parameters: {mount_model.model_param_set}')

    with mounts.make_mount_from_args(args) as mount, \
        laser.make_laser_from_args(args) as laser_pointer, \
        Gamepad() as game_pad:

        telem_logger = telem.make_telem_logger_from_args(args)

        target = targets.make_target_from_args(
            args,
            mount,
            mount_model,
            MeridianSide[args.meridian_side.upper()],
            telem_logger=telem_logger,
            camera_separation_callback=make_camera_separation_callback(
                pid_to_signal=args.pid_to_signal,
                separation_threshold=args.signal_angle,
            ),
        )

        tracker = Tracker(
            mount=mount,
            mount_model=mount_model,
            target=target,
            telem_logger=telem_logger,
        )

        if game_pad is not None:
            if laser_pointer is not None:
                game_pad.register_callback('BTN_SOUTH', laser_pointer.set)
            tracker.register_callback(gamepad_callback)
            if telem_logger is not None:
                telem_logger.register_sources({'gamepad': game_pad})

        stopping_conditions = control.make_stop_conditions_from_args(args)
        tracker.run(stopping_conditions)

if __name__ == "__main__":
    main()
