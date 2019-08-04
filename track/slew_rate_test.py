#!/usr/bin/env python3

import time
import numpy as np
import track

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--mount-type',
        help='select mount type (nexstar or gemini)',
        default='gemini'
    )
    parser.add_argument(
        '--mount-path',
        help='serial device node or hostname for mount command interface',
        default='/dev/ttyACM0'
    )
    args = parser.parse_args()

    # Create object with base type TelescopeMount
    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path)
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    try:
        SLEW_CHANGE_SLEEP = 3.0
        TIME_LIMIT = 60.0
        SLEW_LIMIT = 20.0

        # rates to test in arcseconds per second
        rates = [2**x for x in range(14)]
        rates.append(16319)

        axes = mount.get_axis_names()

        rate_est = {}
        for axis in axes:
            rate_est[axis] = []

        direction = +1

        for axis in axes:
            print('Testing ' + axis + ' axis...')

            if axis == 'ra':
                print('Warning: RA axis results could be inaccurate! RA coordinates change as a'
                    + ' function of time for a stationary mount!')

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
        print('Got CTRL-C, shutting down...')
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...')
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

if __name__ == "__main__":
    main()
