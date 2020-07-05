#!/usr/bin/env python3

"""Direct gamepad control of telescope mount.

This program allows direct control of a telescope mount with a gamepad. The slew rates of the mount
are controlled by the analog sticks.
"""

from threading import Event
import track
from track import laser, mounts, telem

def main():
    """See module docstring"""

    parser = track.ArgParser()
    mounts.add_program_arguments(parser)
    telem.add_program_arguments(parser)
    laser.add_program_arguments(parser)
    args = parser.parse_args()

    mount = mounts.make_mount_from_args(args)

    game_pad = track.Gamepad()

    try:
        laser_pointer = laser.make_laser_from_args(args)
        game_pad.register_callback('BTN_SOUTH', laser_pointer.set)
    except OSError:
        print('Could not connect to laser pointer control device.')
        laser_pointer = None

    if args.telem_enable:
        telem_logger = telem.make_telem_logger_from_args(args, sources={'gamepad': game_pad})
        telem_logger.start()

    try:
        if mount is None:
            # do nothing in this thread until CTRL-C
            Event().wait()
        else:
            while True:
                x, y = game_pad.get_value()
                mount.slew(0, mount.max_slew_rate * x)
                mount.slew(1, mount.max_slew_rate * y)

    except KeyboardInterrupt:
        print('Got CTRL-C, shutting down...')
    finally:
        if mount is not None:
            # don't rely on destructors to safe mount!
            print('Safing mount...', end='', flush=True)
            if mount.safe():
                print('Mount safed successfully!')
            else:
                print('Warning: Mount may be in an unsafe state!')

        if args.telem_enable:
            telem_logger.stop()

        game_pad.stop()

if __name__ == "__main__":
    main()
