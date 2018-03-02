#!/usr/bin/env python

"""Direct gamepad control of telescope mount.

This program allows direct control of a telescope mount with a gamepad. The slew rates of the mount
are controlled by the analog sticks.
"""

from __future__ import print_function
import track

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--scope',
        help='serial device for connection to telescope',
        default='/dev/ttyUSB0'
    )
    parser.add_argument(
        '--bypass-alt-limits',
        help='bypass mount altitude limits',
        action='store_true'
    )
    args = parser.parse_args()

    mount = track.NexStarMount(args.scope, bypass_alt_limits=args.bypass_alt_limits)
    if args.bypass_alt_limits:
        print('Warning: Altitude limits disabled! Be careful!')

    game_pad = track.Gamepad()

    try:
        while True:
            try:
                x, y = game_pad.get_value()
                mount.slew('az', mount.max_slew_rate * x)
                mount.slew('alt', mount.max_slew_rate * y)
            except mount.AxisLimitException:
                pass
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
