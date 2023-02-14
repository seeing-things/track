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
import logging
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
from track import cameras, logs
from track.config import ArgParser, CONFIG_PATH
from track.mounts import MeridianSide
from track.model import (
    save_default_param_set,
    load_stored_param_set,
    apply_guide_cam_alignment_error
)
from track.plate_solve import plate_solve, NoSolutionException


logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT_S = 2.0


def terminate_subprocess(process: subprocess.Popen) -> None:
    """Terminate a running subprocess"""
    logger.info('Terminating subprocess.')
    process.send_signal(signal.SIGINT)
    time_sigint = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - time_sigint > SHUTDOWN_TIMEOUT_S:
            logger.warning('Subprocess shutdown timeout; resorting to SIGKILL.')
            process.kill()
            break
        time.sleep(0.1)
    logger.info('Subprocess terminated.')


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
    parser.add_argument(
        '--non-interactive',
        help='bypass all interactive steps',
        action='store_false',
        dest='interactive'
    )
    cameras.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'align_guidescope')

    # Run the `track` program to get the mount pointed at the approximate location of the star.
    # This is done first so that the spiral search phase doesn't start until the mount is already
    # near the star.
    logger.info(f'Moving mount to the vicinity of {args.star}.')
    subprocess.run(
        args=[
            'track',
            '--no-console-logs',  # confusing to have logs from multiple processes
            f'--meridian-side={args.meridian_side}',
            '--stop-when-converged-angle=0.1',  # stop when within 1 degree of target
            'star',
            args.star,
        ],
        check=True,
    )

    # Run the `track` program as a subprocess. This performs a spiral search until the star is
    # found and then continues to track it to keep it centered in the main telescope camera while
    # the next steps are performed simultaneously.
    logger.info(f'Starting spiral search for {args.star}.')
    # Not using SIG_IGN here since this prevents `sigtimedwait()` from working
    signal.signal(signal.SIGUSR1, lambda *args: None)
    # pylint: disable=consider-using-with
    track_process = subprocess.Popen(
        args=[
            'track',
            '--no-console-logs',  # confusing to have logs from multiple processes
            f'--config-file={args.main_camera_cfg}',  # use the main camera in this subprocess
            f'--meridian-side={args.meridian_side}',
            '--fuse',
            f'--pid-to-signal={os.getpid()}',
            '--spiral-search',
            'star',
            args.star,
        ]
    )
    atexit.register(terminate_subprocess, track_process)

    if args.interactive:
        if not click.confirm(
                f'Is the spiral search completed, with {args.star} centered in the preview window?',
                default=True
            ):
            logger.critical(f'User denies that {args.star} is centered in the preview window.')
            sys.exit(1)
    else:
        # subprocess will send SIGUSR1 to this process when it has converged
        if not signal.sigtimedwait((signal.SIGUSR1,), 120):
            logger.critical('Timeout waiting for mount to converge on the target.')
            sys.exit(1)
        logger.info(f'Converged on {args.star}. Pausing for a moment to let things settle down.')
        time.sleep(5)

    # Solve frame from guidescope camera while the tracking system is keeping the star centered in
    # the main camera.
    logger.info('Plate solving guidescope camera frame.')
    camera = cameras.make_camera_from_args(args)
    frame = camera.get_frame()
    try:
        wcs, _ = plate_solve(
            frame,
            camera_width=camera.field_of_view[1]
        )
    except NoSolutionException:
        logger.critical('Plate solving failed.')
        sys.exit(1)

    # get astrometric geocentric world coordinate of named star
    target = ephem.star(args.star)
    target.compute()
    star_position_eq = SkyCoord(target.a_ra * u.rad, target.a_dec * u.rad)
    pixel_location = wcs.world_to_pixel(star_position_eq)

    # sanity check that the pixel location is actually in the frame
    if not camera.pixel_in_frame(pixel_location[0], pixel_location[1]):
        logger.critical(f'Could not find {args.star} in the frame.')
        sys.exit(1)

    if args.interactive:
        # show the frame annotated with the location of the bright star
        plt.imshow(frame)
        plt.plot(pixel_location[0], pixel_location[1], 'rx')
        plt.title(f'Note whether the red X is on {args.star}, then close this window')
        plt.show()

        if not click.confirm(f'Was the red X on {args.star}?', default=True):
            logger.critical(f'User rejected identification of star {args.star}.')
            sys.exit(1)

    # apply the estimated guidecam alignment error to the stored mount model
    logger.info('Guidescope alignment succeeded! Applying result to saved mount model.')
    guide_cam_align_error = camera.pixel_to_angle(pixel_location[0], pixel_location[1])
    old_param_set = load_stored_param_set()
    new_param_set = apply_guide_cam_alignment_error(old_param_set, guide_cam_align_error)
    save_default_param_set(new_param_set)


if __name__ == "__main__":
    main()
