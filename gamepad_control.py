#!/usr/bin/env python

import config
import configargparse
import mounts
import gamepad
import numpy as np

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--bypass-alt-limits', help='bypass mount altitude limits', action='store_true')
args = parser.parse_args()

mount = mounts.NexStarMount(args.scope, bypass_alt_limits=args.bypass_alt_limits)
if args.bypass_alt_limits:
    print('Warning: Altitude limits disabled! Be careful!')

game_pad = gamepad.Gamepad()

try:
    while True:
        try:
            x, y = game_pad.get_proportional()
            mount.slew('az', mount.max_slew_rate * x)
            mount.slew('alt', mount.max_slew_rate * y)
        except mount.AltitudeLimitException:
            pass
except KeyboardInterrupt:
    pass
