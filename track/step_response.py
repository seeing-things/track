#!/usr/bin/env python3

"""Program for plotting step response of the mount."""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Longitude
import astropy.units as u
from track.config import ArgParser
from track import mounts


def main():
    """Apply step functions of varying magnitudes to a mount axis and plot the responses."""

    parser = ArgParser()
    mounts.add_program_arguments(parser)
    parser.add_argument(
        '--axis',
        help='axis number (0 or 1)',
        default=0,
        type=int,
    )
    args = parser.parse_args()

    step_responses = {}
    direction = +1.0
    with mounts.make_mount_from_args(args) as mount:
        for step_magnitude in np.arange(0.5, 4.5, 0.5):
            positions = []
            position_start = mount.get_position()[args.axis]
            time_start = time.time()
            direction = -direction
            rate = 0.0
            while True:
                delta_position = abs(
                    Longitude(
                        mount.get_position()[args.axis] - position_start, wrap_angle=180 * u.deg
                    )
                )
                time_elapsed = time.time() - time_start
                positions.append({'time': time_elapsed, 'position': delta_position.deg})
                mount.slew(args.axis, rate)
                if time_elapsed >= 1.0:
                    rate = step_magnitude * direction
                if time_elapsed >= 4.0:
                    mount.safe()
                    break
                time.sleep(0.01)

            step_responses[step_magnitude] = pd.DataFrame(positions)

    # plot step responses
    for step_magnitude, response in step_responses.items():
        step_response = np.diff(response.position) / np.diff(response.time)
        plt.plot(response.time[:-1] - 1.0, step_response, label=f'{step_magnitude:.1f} deg/s')

    # plot acceleration limit line
    t = np.linspace(0, 0.5, 1000)
    a = 10 * t  # 10 degrees per second squared -- the default for G11 mount when this was written
    plt.plot(t, a, 'r', label='accel limit')

    plt.title(f'Step Response for Axis {args.axis}')
    plt.xlabel('Time [s]')
    plt.ylabel('Slew Rate [deg/s]')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
