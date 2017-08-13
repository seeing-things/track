#!/usr/bin/env python

import argparse
import mounts
import time
import errorsources
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--backlash-az', help='backlash in azimuth (arcseconds)', default=0.0, type=float)
parser.add_argument('--backlash-alt', help='backlash in altitude (arcseconds)', default=0.0, type=float)
parser.add_argument('--slew-rate', help='slew rate (arcseconds/second)', default=100.0, type=float)
args = parser.parse_args()

if args.backlash_alt > 0 and args.backlash_az > 0:
    print('Can\'t test in both axes simultaneously')
    sys.exit(1)
elif args.backlash_az > 0:
    deadband = args.backlash_az
    axis = 'az'
elif args.backlash_alt > 0:
    deadband = args.backlash_alt
    axis = 'alt'
else:
    print('No positive backlash value specified: set backlash-az or backlash-alt to a positive value')
    sys.exit(1)

print('Testing ' + str(axis) + ' axis')

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

position_start = mount.get_azalt()

while True:
    mount.slew(axis, args.slew_rate / 3600.0)
    while True:
        position = mount.get_azalt()
        position_change = {
            'az': errorsources.wrap_error(position['az'] - position_start['az']) * 3600.0,
            'alt': errorsources.wrap_error(position['alt'] - position_start['alt']) * 3600.0,
        }
        print('Position change (arcseconds): ' + str(position_change))
        if position_change[axis] > deadband:
            break
    mount.slew(axis, -args.slew_rate / 3600.0)
    while True:
        position = mount.get_azalt()
        position_change = {
            'az': errorsources.wrap_error(position['az'] - position_start['az']) * 3600.0,
            'alt': errorsources.wrap_error(position['alt'] - position_start['alt']) * 3600.0,
        }
        print('Position change (arcseconds): ' + str(position_change))
        if position_change[axis] < 0:
            break
