#!/usr/bin/env python3

"""Perform automated alignment of a telescope mount.

This program automates the somewhat tedious process of aligning the mount. It will do the following
things:
1) Generate a list of positions on the sky to point at
2) For each position:
   a) Point the mount at the position
   b) Capture an image with a camera
   c) Use the astrometry.net plate solver to determine the sky coordinates of the image center
   d) Store a timestamp, the mount's encoder positions, and the image sky coordinates
3) Use the set of observations to solve for mount model parameters
4) Store the mount model parameters on disk for future use during the same observing session
"""

import atexit
import logging
import os
import sys
import time
import pickle
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, EarthLocation, Longitude
from track import cameras, gps_client, logs, model, mounts, ntp, telem
from track.cameras import Camera
from track.control import Tracker, smallest_allowed_error
from track.config import ArgParser, CONFIG_PATH, DATA_PATH
from track.model import ModelParamSet, MountModel
from track.mounts import MeridianSide, MountEncoderPositions, TelescopeMount
from track.plate_solve import plate_solve, NoSolutionException
from track.targets import FixedMountEncodersTarget
from track.tsp import Destination, solve_route


logger = logging.getLogger(__name__)


class Position(Destination):
    """A position for use in alignment procedure"""

    def __init__(
        self,
        encoder_positions: MountEncoderPositions,
        mount: TelescopeMount,
    ):
        """
        Args:
            encoder_positions: Mount encoder positions corresponding to this sky position
            mount: Instance of TelescopeMount
        """
        self.encoder_positions = encoder_positions
        self.mount = mount

    def distance_to(self, other_destination: "Position") -> int:
        """Returns the cost of moving the mount from this position to another position

        Assumes that the slew rate is the same for all mount axes.

        Args:
            other_destination: Another instance of this class

        Returns:
            An integer value representing the cost.
        """
        smallest_encoder_errors = []
        for axis, encoder_position in enumerate(self.encoder_positions):
            smallest_encoder_errors.append(
                smallest_allowed_error(
                    encoder_position.deg,
                    other_destination.encoder_positions[axis].deg,
                    self.mount.no_cross_encoder_positions()[axis].deg,
                )
            )
        max_error_mag = np.max(np.abs(smallest_encoder_errors))

        # scale by 1000 to minimize precision loss when quantizing to integer
        return int(1000 * max_error_mag)


def generate_positions(
    min_positions: int,
    mount_model: MountModel,
    mount: TelescopeMount,
    min_altitude: Angle = 0.0 * u.deg,
    meridian_side: Optional[MeridianSide] = None,
) -> List[Position]:
    """Generate a list of equally spaced positions on the sky to search.

    The list of positions generated will be a subset of pixels generated by the HEALPix algorithm,
    which is a method of pixelization of a sphere where each "pixel" has equal area. In the
    context of mount alignment we not so much concerned with the equal area property, but we do
    want positions that are roughly evenly distributed over the sphere, and HEALPix does a
    reasonable job of this as well. The full set of HEALPix pixels is filtered to exclude pixels
    that are below a minimum altitude threshold and (optionally) those that are on the opposite
    side of the meridian. HEALPix is oriented with the top approximately at local zenith. Since
    these positions are used prior to alignment the mount may not point to the actual azimuth
    of each coordinate but this is not important.

    Args:
        min_positions: Minimum number of positions. The actual number of positions returned may be
            larger than requested.
        mount_model: An instance of the mount model.
        mount: A mount interface object.
        min_altitude: Restrict positions to be above this altitude.
        meridian_side: If specified, restricts positions to only be on this side of the meridian,
            where meridian is defined as the great circle passing through local zenith and the
            mount pole.

    Returns:
        A list of Position objects to use for alignment.
    """
    level = 0
    while True:
        healpix = HEALPix(nside=2**level)
        positions = []
        for i in range(healpix.npix):
            # interpret each HEALPix as a topocentric (AzAlt) coordinate
            (az, alt) = healpix.healpix_to_lonlat(i)

            # skip points on wrong side of mount meridian
            if az > 180 * u.deg:  # west of meridian
                if meridian_side == MeridianSide.EAST:
                    continue
                side = MeridianSide.WEST
            if az <= 180 * u.deg:  # east of meridian
                if meridian_side == MeridianSide.WEST:
                    continue
                side = MeridianSide.EAST

            # skip points below min altitude threshold
            if alt < min_altitude:
                continue

            position_topo = SkyCoord(az, alt, frame='altaz')
            encoder_positions = mount_model.topocentric_to_encoders(position_topo, side)
            positions.append(
                Position(
                    encoder_positions=encoder_positions,
                    mount=mount,
                )
            )

        if len(positions) >= min_positions:
            break

        # not enough positions -- try again with a higher level HEALPix
        level += 1

    return positions


def attempt_plate_solving(
    camera: Camera,
    mount: TelescopeMount,
    location: EarthLocation,
    observations: List,
    observations_dir: str,
    num_solutions_so_far: int,
) -> bool:
    """Attempt to do plate solving.

    Args:
        camera: A frame will be captured from this camera to use for plate solving.
        mount: The mount.
        location: Observer location. Used to transform equatorial coordinate to topocentric.
        observations: List of observations. If plate solving is successful a new dict is appended.
        observations_dir: Directory where the FITS image will be saved if a solution is found.
        num_solutions_so_far: Number of successful solutions up to this point in the alignment
            procedure. Used to generate unique filename for FITS image.

    Returns:
        True on success, False on failure.
    """

    # Make all physical measurements (clock, camera, mount encoders) near same time
    timestamp = time.time()
    frame = camera.get_frame()
    mount_position = mount.get_position()

    try:
        _, sc_eq = plate_solve(frame, camera_width=camera.field_of_view[1])
    except NoSolutionException:
        return False
    else:
        sc_eq.obstime = Time(timestamp, format='unix')
        sc_eq.location = location
        sc_topo = sc_eq.transform_to('altaz')
        observations.append(
            {
                'unix_timestamp': timestamp,
                'encoder_0': mount_position.encoder_0.deg,
                'encoder_1': mount_position.encoder_1.deg,
                'sky_az': sc_topo.az.deg,
                'sky_alt': sc_topo.alt.deg,
                'sky_ra': sc_eq.ra.deg,
                'sky_dec': sc_eq.dec.deg,
            }
        )
        hdu = fits.PrimaryHDU(frame)
        try:
            hdu.writeto(
                os.path.join(observations_dir, f'camera_frame_{num_solutions_so_far:03d}.fits')
            )
        except OSError:
            logger.exception('Trouble saving image to disk.')
        return True


def remove_empty_dir(directory: str) -> None:
    """Remove a directory if it exists and is empty."""
    try:
        os.rmdir(directory)
    except (OSError, NameError):
        pass


def main():
    """Run the alignment procedure! See module docstring for a description."""

    parser = ArgParser(additional_config_files=[os.path.join(CONFIG_PATH, 'align.cfg')])
    align_group = parser.add_argument_group(
        title='Alignment Configuration',
        description='Options that apply to alignment',
    )
    align_group.add_argument(
        '--mount-pole-alt',
        required=True,
        help='altitude of mount pole above horizon (deg)',
        type=float,
    )
    align_group.add_argument(
        '--guide-cam-orientation',
        help='orientation of guidescope camera, clockwise positive (deg)',
        default=0.0,
        type=float,
    )
    align_group.add_argument(
        '--min-positions',
        help='minimum number of positions to add to mount alignment model',
        default=10,
        type=int,
    )
    align_group.add_argument(
        '--timeout',
        help='max time to wait for mount to converge on a new position',
        default=120.0,
        type=float,
    )
    align_group.add_argument(
        '--max-tries',
        help='max number of plate solving attempts at each position',
        default=3,
        type=int,
    )
    align_group.add_argument(
        '--min-alt',
        help='minimum altitude of alignment positions in degrees',
        default=20.0,
        type=float,
    )
    align_group.add_argument(
        '--max-rms-error',
        help='warning printed if RMS error in degrees is greater than this',
        default=0.2,
        type=float,
    )
    gps_client.add_program_arguments(parser)
    mounts.add_program_arguments(parser)
    cameras.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    ntp.add_program_arguments(parser)
    telem.add_program_arguments(parser)
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'align')

    # Check if system clock is synchronized to GPS
    try:
        ntp.check_ntp_status()
    except ntp.NTPCheckFailure as e:
        if args.ignore_ntp_check:
            logger.warn(f'NTP check failed: {e}')
        else:
            logger.critical(f'NTP check failed: {e}')
            sys.exit(1)

    # Get location of observer from arguments or from GPS
    location = gps_client.make_location_from_args(args)

    # This is meant to be just good enough that the model is able to tell roughly which direction
    # is up so that the positions used during alignment are all above the horizon.
    starter_mount_model = model.load_default_model(
        mount_pole_alt=Longitude(args.mount_pole_alt * u.deg), location=location
    )

    camera = cameras.make_camera_from_args(args)
    telem_logger = telem.make_telem_logger_from_args(args)

    if args.meridian_side is not None:
        meridian_side = MeridianSide[args.meridian_side.upper()]
    else:
        meridian_side = None

    with mounts.make_mount_from_args(args) as mount:
        positions = generate_positions(
            min_positions=args.min_positions,
            mount_model=starter_mount_model,
            mount=mount,
            min_altitude=Angle(args.min_alt * u.deg),
            meridian_side=meridian_side,
        )
        logger.info(f'Generated {len(positions)} positions.')

        # add the park position to the start of the list to serve as the "depot" in the route
        park_encoder_positions = mount.get_position()
        positions.insert(
            0,
            Position(
                encoder_positions=park_encoder_positions,
                mount=mount,
            ),
        )

        # use travelling salesman solver to sort positions in the order that takes the least time
        positions = solve_route(positions)

        # mount is already at the starting position so don't need to go there
        del positions[0]

        # directory in which to place observation data for debugging purposes
        observations_dir = os.path.join(
            DATA_PATH,
            'alignment_' + datetime.utcnow().isoformat(timespec='seconds').replace(':', ''),
        )
        os.makedirs(DATA_PATH, exist_ok=True)
        os.mkdir(observations_dir)
        atexit.register(remove_empty_dir, observations_dir)

        observations = []
        num_solutions = 0
        for idx, position in enumerate(positions):
            target = FixedMountEncodersTarget(
                position.encoder_positions,
                starter_mount_model,
            )

            pos = target.get_position()
            logger.info(
                f'Moving to position {idx + 1} of {len(positions)}: '
                f'Az: {pos.topo.az.deg:.2f}, Alt: {pos.topo.alt.deg:.2f}'
            )

            tracker = Tracker(
                mount=mount,
                mount_model=starter_mount_model,
                target=target,
                telem_logger=telem_logger,
            )
            stop_reason = tracker.run(
                tracker.StoppingConditions(timeout=args.timeout, error_threshold=Angle(2.0 * u.deg))
            )
            mount.safe()
            if tracker.StopReason.CONVERGED not in stop_reason:
                raise RuntimeError(f'Unexpected tracker stop reason: {stop_reason}')
            time.sleep(1.0)

            logger.info(f'Converged on position {idx + 1}. Attempting plate solving.')

            # plate solver doesn't always work on the first try
            for i in range(args.max_tries):
                logger.info(f'Plate solver attempt {i + 1} of {args.max_tries}.')

                if attempt_plate_solving(
                    mount=mount,
                    camera=camera,
                    location=location,
                    observations=observations,
                    observations_dir=observations_dir,
                    num_solutions_so_far=num_solutions,
                ):
                    logger.info(
                        f'Plate solver solution found on attempt {i + 1} of {args.max_tries}.'
                    )
                    num_solutions += 1
                    break

                logger.warning(f'Plate solver failed on attempt {i + 1} of {args.max_tries}.')

    logger.info(f'Plate solver found solutions at {num_solutions} of {len(positions)} positions.')

    if num_solutions == 0:
        logger.critical('Plate solver failed at all positions.')
        sys.exit(1)
    elif num_solutions < args.min_positions:
        logger.warning(
            f'Only {num_solutions} positions usable but {args.min_positions} were required.'
        )

    observations = pd.DataFrame(observations)

    observations_filename = os.path.join(observations_dir, 'observations_dataframe.pickle')
    logger.info(f'Saving observations to {observations_filename}')
    with open(observations_filename, 'wb') as f:
        pickle.dump(observations, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Solving for mount model parameters.')
    model_params, result = model.solve_model(observations)
    logger.info(result)

    rms_error = np.sqrt(2 * result.cost / len(observations))
    if rms_error > args.max_rms_error:
        logger.warning(
            f'Model solution RMS error {rms_error:.4f} > {args.max_rms_error:.4f} degrees'
        )
    else:
        logger.info(f'Model solution RMS error: {rms_error:.4f} degrees')

    model_param_set = ModelParamSet(
        model_params=model_params,
        guide_cam_orientation=Longitude(args.guide_cam_orientation * u.deg),
        location=location,
        timestamp=time.time(),
    )

    params_filename = os.path.join(observations_dir, 'model_params.pickle')
    logger.info(f'Saving model parameters to {params_filename}')
    with open(params_filename, 'wb') as f:
        pickle.dump(model_param_set, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Making this set of model parameters the default.')
    model.save_default_param_set(model_param_set)


if __name__ == "__main__":
    main()
