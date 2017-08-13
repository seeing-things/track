#!/usr/bin/env python

import config
import configargparse
import mounts
import errorsources

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

position_start = mount.get_azalt()

while True:
    position = mount.get_azalt()
    position_change = {
        'az': errorsources.wrap_error(position['az'] - position_start['az']) * 3600.0,
        'alt': errorsources.wrap_error(position['alt'] - position_start['alt']) * 3600.0,
    }
    print(str(position_change))
