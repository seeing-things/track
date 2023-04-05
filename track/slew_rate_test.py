#!/usr/bin/env python3

"""Test to compare commanded slew rates to actual slew rates."""

import time
import astropy.units as u
from astropy.coordinates import Longitude
from track import config, mounts

SLEW_CHANGE_TIME = 3.0
TIME_LIMIT = 60.0
SLEW_LIMIT = 20.0


def main():
    """See module docstring."""
    parser = config.ArgParser()
    mounts.add_program_arguments(parser)
    args = parser.parse_args()

    # rates to test in arcseconds per second
    rates = [2**x for x in range(14)]
    rates.append(16319)
    rate_est = {}

    with mounts.make_mount_from_args(args) as mount:
        axes = mount.AxisName

        direction = +1

        for axis in axes:
            print(f'Testing {axis}.')
            rate_est[axis] = []

            for rate in rates:
                print(f'Commanding slew at {rate} arcseconds per second.')

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
                        Longitude(position[axis] - position_start[axis], wrap_angle=180 * u.deg)
                    ).deg
                    if position_change > SLEW_LIMIT or time_elapsed > TIME_LIMIT:
                        break

                rate_est[axis].append(position_change / time_elapsed)
                print(f'\tmeasured rate: {rate_est[axis][-1] * 3600.0}')

    print('Results:')
    for rate, rate_est_0, rate_est_1 in zip(rates, rate_est[axes[0]], rate_est[axes[1]]):
        print(f'{rate}, {3600 * rate_est_0}, {3600 * rate_est_1}')


if __name__ == "__main__":
    main()
