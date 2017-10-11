#!/usr/bin/env python

import config
import configargparse
import mounts
import errorsources
import time
import numpy as np
import csv

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES, formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--backlash-az', help='backlash in azimuth (arcseconds)', default=0.0, type=float)
parser.add_argument('--backlash-alt', help='backlash in altitude (arcseconds)', default=0.0, type=float)
parser.add_argument('--align-dir-az', help='azimuth alignment approach direction (-1 or +1)', default=+1, type=int)
parser.add_argument('--align-dir-alt', help='altitude alignment approach direction (-1 or +1)', default=+1, type=int)
args = parser.parse_args()

mount = mounts.NexStarMount(args.scope)
mount.set_backlash('az', args.align_dir_az, args.backlash_az / 3600.0)
mount.set_backlash('alt', args.align_dir_alt, args.backlash_alt / 3600.0)

SLEW_RANGE_DEG = 1.0
SLEW_RATE = 0.1
PERIODS = 3

try:

    direction_start = +1
    for axis in ['az', 'alt']:
        print('Testing ' + axis + ' axis...')
        filename = axis + '.csv'
        with open(filename, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            position_start = mount.get_azalt()
            time_start = time.time()
            for period in range(PERIODS):
                print('Period ' + str(period + 1) + ' of ' + str(PERIODS))
                print('Starting positive ' + axis + ' slew...')
                mount.slew(axis, SLEW_RATE)
                while True:
                    position = mount.get_azalt()
                    position_corrected = mount.remove_backlash(position, {'az':True, 'alt':True})
                    position_change = errorsources.wrap_error(position[axis] - position_start[axis])
                    position_change_corrected = errorsources.wrap_error(position_corrected[axis] - position_start[axis])
                    time_elapsed = time.time() - time_start
                    csvwriter.writerow([
                        time_elapsed,
                        position[axis] * 3600.0,
                        position_change * 3600.0,
                        position_change_corrected * 3600.0
                    ])
                    if abs(position_change) > SLEW_RANGE_DEG:
                        break
                print('Starting negative ' + axis + ' slew...')
                mount.slew(axis, -SLEW_RATE)
                position_change_prev = position_change
                while True:
                    position = mount.get_azalt()
                    position_corrected = mount.remove_backlash(position, {'az':True, 'alt':True})
                    position_change = errorsources.wrap_error(position[axis] - position_start[axis])
                    position_change_corrected = errorsources.wrap_error(position_corrected[axis] - position_start[axis])
                    time_elapsed = time.time() - time_start
                    csvwriter.writerow([
                        time_elapsed,
                        position[axis] * 3600.0,
                        position_change * 3600.0,
                        position_change_corrected * 3600.0
                    ])
                    if np.sign(position_change) != np.sign(position_change_prev):
                        break
                    position_change_prev = position_change
        mount.slew(axis, 0.0)

except KeyboardInterrupt:
    print('Goodbye!')
    pass
