#!/usr/bin/env python

import mounts
import errorsources
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

try:
    SLEW_CHANGE_SLEEP = 3.0
    TEST_DURATION = 5.0

    # rates to test in arcseconds per second
    rates = [2**x for x in range(14)]
    rates.append(16319)

    rate_est = {'az': [], 'alt': []}

    direction = +1

    for axis in ['az', 'alt']:
        print('Testing ' + axis + ' axis...')
        for rate in rates:

            print('Commanding slew at ' + str(rate) + ' arcseconds per second...')

            mount.slew(axis, direction * rate / 3600.0)
            direction *= -1
            time.sleep(SLEW_CHANGE_SLEEP)
            position_start = mount.get_azalt()
            time_start = time.time()
            time.sleep(TEST_DURATION)
            position_stop = mount.get_azalt()
            time_elapsed = time.time() - time_start

            rate_est[axis].append(abs(errorsources.wrap_error(position_stop[axis] - position_start[axis])) / time_elapsed)

        mount.slew(axis, 0.0)

    print('Results:')
    for rate, rate_est_az, rate_est_alt in zip(rates, rate_est['az'], rate_est['alt']):
        print(str(rate) + ', ' + str(3600 * rate_est_az) + ', ' + str(3600 * rate_est_alt))


except KeyboardInterrupt:
    print('Goodbye!')
    pass
