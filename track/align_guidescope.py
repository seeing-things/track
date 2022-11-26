#!/usr/bin/env python3

"""
Run a procedure to estimate the angular offset between the guidescope camera and the main telescope
camera. If this is completed successfully the estimated offset is saved with the active mount
alignment model. The goal is to eliminate the need for ultra-precise physical alignment of the
guidescope, which can be challenging and time consuming. With software-based guidescope alignment,
the physical alignment only needs to be such that the main telescope camera is within the
guidescope camera's field of view.
"""

import atexit
import os
import sys
import signal
import subprocess
import time
from astropy.coordinates import SkyCoord
import astropy.units as u
import click
import ephem
import matplotlib.pyplot as plt
import track
from track import cameras
from track.config import ArgParser, CONFIG_PATH
from track.mounts import MeridianSide
from track.model import (
    save_default_param_set,
    load_stored_param_set,
    apply_guide_cam_alignment_error
)


SHUTDOWN_TIMEOUT_S = 2.0


def terminate_subprocess(process: subprocess.Popen) -> None:
    """Terminate a running subprocess"""
    print('Terminating subprocess...', end='', flush=True)
    process.send_signal(signal.SIGINT)
    time_sigint = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - time_sigint > SHUTDOWN_TIMEOUT_S:
            print('Subprocess shutdown timeout; resorting to SIGKILL...', end='', flush=True)
            process.kill()
            break
        time.sleep(0.1)
    print('done.')


def main():
    """See module docstring at the top of this file."""
    parser = ArgParser(additional_config_files=[os.path.join(CONFIG_PATH, 'align.cfg')])
    parser.add_argument('star', help='Name of bright star')
    parser.add_argument(
        '--main-camera-cfg',
        help='Configuration file with settings for the main camera',
        default=os.path.join(CONFIG_PATH, 'track_main_cam.cfg')
    )
    parser.add_argument(
        '--meridian-side',
        help='side of meridian for equatorial mounts to prefer',
        required=True,
        choices=tuple(m.name.lower() for m in MeridianSide),
    )
    cameras.add_program_arguments(parser)
    args = parser.parse_args()


    # Run the `track` program to get the mount pointed at the approximate location of the star.
    print(f'Moving mount to the vicinity of {args.star}...', end='', flush=True)
    subprocess.run(
        args=[
            'track',
            f'--meridian-side={args.meridian_side}',
            '--stop-when-converged-angle=0.1',  # stop when within 1 degree of target
            'star',
            args.star,
        ],
        check=True,
    )
    print('done.')

    # Run the `track` program as a subprocess. This performs a spiral search until the star is
    # found and then continues to track it to keep it centered in the main telescope camera while
    # the next steps are performed simultaneously.
    print(f'Starting spiral search for {args.star}. Wait until the star is centered in the preview '
        'window before answering the question below.')
    track_process = subprocess.Popen(
        args=[
            'track',
            f'--config-file={args.main_camera_cfg}',  # use the main camera in this subprocess
            f'--meridian-side={args.meridian_side}',
            '--fuse',
            '--spiral-search',
            'star',
            args.star,
        ]
    )
    atexit.register(terminate_subprocess, track_process)

    if not click.confirm(
            f'Is the spiral search completed, with {args.star} centered in the preview window?',
            default=True
        ):
        print('Aborting')
        sys.exit(1)

    # Solve frame from guidescope camera while the tracking system is keeping the star centered in
    # the main camera.
    print('Plate solving guidescope camera frame...', end='', flush=True)
    camera = cameras.make_camera_from_args(args)
    frame = camera.get_frame()
    wcs, _ = track.plate_solve(
        frame,
        camera_width=camera.field_of_view[1]
    )
    print('done.')

    # get astrometric geocentric world coordinate of named star
    target = ephem.star(args.star)
    target.compute()
    star_position_eq = SkyCoord(target.a_ra * u.rad, target.a_dec * u.rad)
    pixel_location = wcs.world_to_pixel(star_position_eq)

    # sanity check that the pixel location is actually in the frame
    if not camera.pixel_in_frame(pixel_location[0], pixel_location[1]):
        print(f'Could not find {args.star} in the frame')
        sys.exit(1)

    # show the frame annotated with the location of the bright star
    plt.imshow(frame)
    plt.plot(pixel_location[0], pixel_location[1], 'rx')
    plt.title(f'Note whether the red X is on {args.star}, then close this window')
    plt.show()

    if not click.confirm(f'Was the red X on {args.star}?', default=True):
        print('Aborting')
        sys.exit(1)

    # apply the estimated guidecam alignment error to the stored mount model
    print('Alignment succeeded! Applying result to saved mount model...', end='', flush=True)
    guide_cam_align_error = camera.pixel_to_angle(pixel_location[0], pixel_location[1])
    old_param_set = load_stored_param_set()
    new_param_set = apply_guide_cam_alignment_error(old_param_set, guide_cam_align_error)
    save_default_param_set(new_param_set)
    print('done.')


if __name__ == "__main__":
    main()
