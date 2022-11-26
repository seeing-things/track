#!/usr/bin/env python3

"""Test to compare commanded slew rates to actual slew rates."""

import sys
import time
import astropy.units as u
from astropy.coordinates import Longitude
from track.config import ArgParser
from track.mounts import LosmandyGeminiMount, NexStarMount

SLEW_CHANGE_TIME = 3.0
TIME_LIMIT = 60.0
SLEW_LIMIT = 20.0

def main():
    """See module docstring"""

    parser = ArgParser()
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
        mount = NexStarMount(args.mount_path)
    elif args.mount_type == 'gemini':
        mount = LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    axes = mount.AxisName

    # pylint: disable=too-many-nested-blocks
    try:
        # rates to test in arcseconds per second
        rates = [2**x for x in range(14)]
        rates.append(16319)

        rate_est = {}
        for axis in axes:
            rate_est[axis] = []

        direction = +1

        for axis in axes:
            print('Testing ' + str(axis) + '...')

            for rate in rates:

                print('Commanding slew at ' + str(rate) + ' arcseconds per second...')

                time_start = time.time()
                while time.time() - time_start < SLEW_CHANGE_TIME:
                    mount.slew(axis, direction * rate / 3600.0)

                direction *= -1
                position_start = mount.get_position()
                time_start = time.time()
                while True:
                    position = mount.get_position()
                    time_elapsed = time.time() - time_start
                    position_change = abs(
                        Longitude(position[axis] - position_start[axis], wrap_angle=180*u.deg)
                    ).deg
                    if position_change > SLEW_LIMIT or time_elapsed > TIME_LIMIT:
                        break

                rate_est[axis].append(position_change / time_elapsed)
                print('\tmeasured rate: ' + str(rate_est[axis][-1] * 3600.0))

            mount.safe()

        print('Results:')
        for rate, rate_est_0, rate_est_1 in zip(rates, rate_est[axes[0]], rate_est[axes[1]]):
            print(str(rate) + ', ' + str(3600 * rate_est_0) + ', ' + str(3600 * rate_est_1))


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
