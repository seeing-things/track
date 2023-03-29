#!/usr/bin/env python3

"""
Does all of the following:

1. Alignment.
2. Guidescope alignment.
3. Autofocus.
"""

import atexit
import logging
import subprocess
import sys
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import skyfield.api
from skyfield.positionlib import Barycentric
from track import logs, gps_client, model, mounts, stars
from track.config import ArgParser
from track.model import MountModel
from track.mounts import MeridianSide, TelescopeMount
from track.subprogram import terminate_subprocess
from track.skyfield_utils import astropy_to_skyfield_observer


logger = logging.getLogger(__name__)


def pick_meridian_side(
    observer: Barycentric,
    star: skyfield.api.Star,
    mount_model: MountModel,
    mount: TelescopeMount,
) -> MeridianSide | None:
    """Automatically pick which meridian side to use.

    If the star is reachable from either meridian side this function makes an arbitrary choice.
    This could be problematic if the star is near an axis limit and also headed towards the limit.
    No attempt is made to detect or avoid that case.

    Args:
        observer: Represents the location and time of the observer.
        star: The star or other object to be observed.
        mount_model: Model to transform between equatorial and mount encoder coordinates.
        mount: Mount object used to test reachability of the sky location.

    Returns:
        A `MeridianSide` if the star is reachable or None if the location is unreachable from
        either side. If the star is reachable from both sides then one is picked arbitrarily.
    """
    alt, az, _ = observer.observe(star).apparent().altaz()
    star_coord = SkyCoord(az.degrees * u.deg, alt.degrees * u.deg, frame='altaz')
    for side in MeridianSide:
        encoder_pos = mount_model.topocentric_to_encoders(star_coord, side)
        if mount.reachable(encoder_pos):
            return side

    return None


def run_program(args: list[str]) -> None:
    """Run a program as a subprocess, blocking until it ends.

    This is similar to `subprocess.run()` but doesn't send SIGKILL to the process when exceptions
    are raised in this process while waiting for the process to end. The most likely source of
    such an exception would be a `KeyboardInterrupt` or `SystemExit` exception raised in response
    to a signal like `SIGINT` in this parent process.

    If the process exit code is nonzero the program quits with exit code 1.

    Args:
        args: List of command line arguments passed to `subprocess.Popen()`. The first item is
            used as the program name in log messages.
    """
    program_name = args[0]
    logger.info(f'Running program: {program_name}')
    # subprocess.run() sends SIGKILL to process on exceptions so do not use that here
    # pylint: disable=consider-using-with
    process = subprocess.Popen(args)
    atexit.register(terminate_subprocess, process)
    if retcode := process.wait():
        logger.critical(f'{program_name} program exited with code {retcode}.')
        sys.exit(1)


def main():
    """See module docstring."""

    parser = ArgParser()
    logs.add_program_arguments(parser)
    gps_client.add_program_arguments(parser)
    mounts.add_program_arguments(parser)
    stars.add_program_arguments(parser)
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'bootstrap')

    location = gps_client.make_location_from_args(args)
    observer = astropy_to_skyfield_observer(location, Time.now())
    star = stars.make_star_from_args(args, observer)
    logger.info(f'Selected star {star.names[0]} for guidescope alignment and autofocus.')

    run_program(['align'])

    with mounts.make_mount_from_args(args) as mount:
        meridian_side = pick_meridian_side(
            observer=observer,
            star=star,
            mount=mount,
            mount_model=model.load_stored_model(max_age=None),
        )

    if meridian_side is None:
        logger.critical(f'{star.names[0]} is not reachable from either meridian side.')
        sys.exit(1)
    logger.info(f'Using meridian side {meridian_side.name}.')

    run_program(
        [
            'align_guidescope',
            '--non-interactive',
            f'--meridian-side={meridian_side.name}',
            'coord-eq',
            # pylint: disable=protected-access
            str(star.ra._degrees),
            str(star.dec.degrees),
        ]
    )

    run_program(
        [
            'autofocus',
            f'--meridian-side={meridian_side.name}',
            'coord-eq',
            # pylint: disable=protected-access
            str(star.ra._degrees),
            str(star.dec.degrees),
        ]
    )

    logger.info('All setup steps completed. Ready to track targets!')


if __name__ == "__main__":
    main()
