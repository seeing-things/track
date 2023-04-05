#!/usr/bin/env python3

"""Direct gamepad control of telescope mount.

This program allows direct control of a telescope mount with a gamepad. The slew rates of the mount
are controlled by the analog sticks.
"""

import logging
import os
import signal
import sys
from track.config import ArgParser
from track.gamepad import Gamepad
from track import laser, logs, mounts, telem


logger = logging.getLogger(__name__)


def main():
    """See module docstring."""
    parser = ArgParser()
    mounts.add_program_arguments(parser)
    telem.add_program_arguments(parser)
    laser.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'gamepad_control')

    with (
        mounts.make_mount_from_args(args, use_multiprocessing=False) as mount,
        laser.make_laser_from_args(args) as laser_pointer,
        telem.make_telem_logger_from_args(args) as telem_logger,
        Gamepad() as game_pad,
    ):
        if game_pad is None:
            print('This program requires a gamepad but none were found.')
            sys.exit(1)

        if telem_logger is not None:
            telem_logger.register_sources({'gamepad': game_pad})

        if laser_pointer is not None:
            game_pad.register_callback('BTN_SOUTH', laser_pointer.set)

        # Quit when Start button is pressed (called from another thread so can't use sys.exit)
        game_pad.register_callback('BTN_START', lambda _: os.kill(os.getpid(), signal.SIGINT))

        # Directly control mount slew rates with gamepad
        while True:
            x, y = game_pad.get_value()
            mount.slew(0, mount.max_slew_rate * x)
            mount.slew(1, mount.max_slew_rate * y)


if __name__ == "__main__":
    main()
