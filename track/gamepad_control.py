#!/usr/bin/env python

import track

parser = track.ArgParser()
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--bypass-alt-limits', help='bypass mount altitude limits', action='store_true')
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
        except mount.AltitudeLimitException:
            pass
except KeyboardInterrupt:
    pass
