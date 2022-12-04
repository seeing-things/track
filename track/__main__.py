#!/usr/bin/env python3

"""Automated tracking of objects.

This is the core program in this project. The software uses a control system to cause a telescope
mount to track a target based on the target's position in the sky and the mount's encoder readings.
"""
import logging
import os
import sys
import click
import astropy.units as u
from astropy.coordinates import Longitude
from track import control, laser, logs, model, mounts, ntp, targets, telem
from track.config import ArgParser, CONFIG_PATH
from track.control import Tracker
from track.gamepad import Gamepad
from track.mounts import MeridianSide


logger = logging.getLogger(__name__)


def main():
    """See module docstring"""

    def gamepad_callback(_tracker: Tracker) -> bool:
        """Callback for gamepad control.

        Allows manual control of the slew rate via a gamepad when the 'B' button is held down,
        overriding tracking behavior. This callback is registered with the Tracker object which
        calls it on each control cycle.

        Defined inside main() so this has easy access to objects that are within that scope.

        Args:
            _tracker: A reference to an object of type Tracker. Not used internally.

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
    args = parser.parse_args()

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
