#!/usr/bin/env python

import time
import numpy as np
import track

def main():

    parser = track.ArgParser()
    parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
    args = parser.parse_args()

    # Create object with base type TelescopeMount
    mount = track.NexStarMount(args.scope)

    try:
        SLEW_CHANGE_SLEEP = 3.0
        TIME_LIMIT = 60.0
        SLEW_LIMIT = 20.0

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
                position_start = mount.get_position()
                time_start = time.time()
                while True:
                    position = mount.get_position()
                    time_elapsed = time.time() - time_start
                    position_change = abs(track.wrap_error(position[axis] - position_start[axis]))
                    if position_change > SLEW_LIMIT or time_elapsed > TIME_LIMIT:
                        break

                rate_est[axis].append(position_change / time_elapsed)
                print('\tmeasured rate: ' + str(rate_est[axis][-1] * 3600.0))

            mount.slew(axis, 0.0)

        print('Results:')
        for rate, rate_est_az, rate_est_alt in zip(rates, rate_est['az'], rate_est['alt']):
            print(str(rate) + ', ' + str(3600 * rate_est_az) + ', ' + str(3600 * rate_est_alt))


    except KeyboardInterrupt:
        print('Goodbye!')
        pass

if __name__ == "__main__":
    main()
