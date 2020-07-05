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

import os
import sys
import time
import pickle
from datetime import datetime
from typing import List, Optional, NamedTuple
import numpy as np
import pandas as pd
import click
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle, Longitude
import track
from track import cameras, mounts, telem
from track.control import Tracker
from track.config import DATA_PATH
from track.model import ModelParamSet
from track.mounts import MeridianSide
from track.targets import FixedTopocentricTarget
from track.gps_client import GPSValues, GPSMargins, GPS


class Position(NamedTuple):
    """A position for use in alignment procedure

    Attributes:
        position: SkyCoord in topocentric "altaz" frame.
        meridian_side: Side of meridian mount should use for this position.
    """
    position: SkyCoord
    meridian_side: MeridianSide


def generate_positions(
        min_positions: int,
        min_altitude: Angle = 0.0*u.deg,
        meridian_side: Optional[MeridianSide] = None
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
            if az > 180*u.deg:  # west of meridian
                if meridian_side == MeridianSide.EAST:
                    continue
                side = MeridianSide.WEST
            if az <= 180*u.deg:  # east of meridian
                if meridian_side == MeridianSide.WEST:
                    continue
                side = MeridianSide.EAST

            # skip points below min altitude threshold
            if alt < min_altitude:
                continue

            positions.append(Position(SkyCoord(az, alt, frame='altaz'), side))

        if len(positions) >= min_positions:
            break

        # not enough positions -- try again with a higher level HEALPix
        level += 1

    return positions


def main():
    """Run the alignment procedure! See module docstring for a description."""

    parser = track.ArgParser()
    align_group = parser.add_argument_group(
        title='Alignment Configuration',
        description='Options that apply to alignment',
    )
    align_group.add_argument(
        '--mount-pole-alt',
        required=True,
        help='altitude of mount pole above horizon (deg)',
        type=float
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
        type=int
    )
    align_group.add_argument(
        '--timeout',
        help='max time to wait for mount to converge on a new position',
        default=120.0,
        type=float
    )
    align_group.add_argument(
        '--max-tries',
        help='max number of plate solving attempts at each position',
        default=3,
        type=int
    )
    align_group.add_argument(
        '--min-alt',
        help='minimum altitude of alignment positions in degrees',
        default=20.0,
        type=float
    )
    align_group.add_argument(
        '--max-rms-error',
        help='warning printed if RMS error in degrees is greater than this',
        default=0.2,
        type=float
    )
    observer_group = parser.add_argument_group(
        title='Observer Location Options',
        description='Setting all three of these options will override GPS',
    )
    observer_group.add_argument(
        '--lat',
        help='latitude of observer (+N)',
        type=float,
    )
    observer_group.add_argument(
        '--lon',
        help='longitude of observer (+E)',
        type=float,
    )
    observer_group.add_argument(
        '--elevation',
        help='elevation of observer (m)',
        type=float
    )
    mounts.add_program_arguments(parser)
    cameras.add_program_arguments(parser, profile='align')
    telem.add_program_arguments(parser)
    args = parser.parse_args()

    mount = mounts.make_mount_from_args(args)

    # Get location of observer from arguments or from GPS
    if all(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        print('Location (lat, lon, elevation) specified by program args. This will override GPS.')
        if not click.confirm('Proceed without GPS location?', default=True):
            sys.exit(1)
        location = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.elevation*u.m)
    elif any(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        parser.error("Must give all of lat, lon, and elevation or none of them.")
    else:
        with GPS() as g:
            location = g.get_location(
                timeout=10.0,
                need_3d=True,
                err_max=GPSValues(
                    lat=100.0,
                    lon=100.0,
                    alt=100.0,
                    track=np.inf,
                    speed=np.inf,
                    climb=np.inf,
                    time=0.01
                ),
                margins=GPSMargins(speed=np.inf, climb=np.inf, time=1.0)
            )
            print(
                'Got location from GPS: '
                f'lat: {location.lat:.5f}, '
                f'lon: {location.lon:.5f}, '
                f'altitude: {location.height:.2f}'
            )

    # This is meant to be just good enough that the model is able to tell roughly which direction
    # is up so that the positions used during alignment are all above the horizon.
    starter_mount_model = track.model.load_default_model(
        mount_pole_alt=Longitude(args.mount_pole_alt*u.deg),
        location=location
    )

    camera = cameras.make_camera_from_args(args, profile='align')

    if args.telem_enable:
        telem_logger = telem.make_telem_logger_from_args(args)
        telem_logger.start()

    if args.meridian_side is not None:
        meridian_side = MeridianSide[args.meridian_side.upper()]
    else:
        meridian_side = None
    positions = generate_positions(
        min_positions=args.min_positions,
        min_altitude=Angle(args.min_alt*u.deg),
        meridian_side=meridian_side
    )

    # directory in which to place observation data for debugging purposes
    observations_dir = os.path.join(
        DATA_PATH,
        'alignment_' + datetime.utcnow().isoformat(timespec='seconds').replace(':', '')
    )
    os.mkdir(observations_dir)

    # pylint: disable=broad-except
    try:
        observations = []
        num_solutions = 0
        for idx, position in enumerate(positions):

            print(f'Moving to position {idx + 1} of {len(positions)}: '
                  f'Az: {position.position.az.deg:.2f}, Alt: {position.position.alt.deg:.2f}')

            target = FixedTopocentricTarget(
                position.position,
                starter_mount_model,
                position.meridian_side
            )
            tracker = Tracker(
                mount=mount,
                mount_model=starter_mount_model,
                target=target
            )
            telem_logger.sources['tracker'] = tracker
            stop_reason = tracker.run(tracker.StoppingConditions(
                timeout=args.timeout, error_threshold=Angle(2.0 * u.deg)
            ))
            mount.safe()
            if tracker.StopReason.CONVERGED not in stop_reason:
                raise RuntimeError(f'Unexpected tracker stop reason: {stop_reason}')
            time.sleep(1.0)

            print('Converged on the target position. Attempting plate solving.')

            # plate solver doesn't always work on the first try
            for i in range(args.max_tries):

                print(f'\tPlate solver attempt {i + 1} of {args.max_tries}...', end='', flush=True)

                # Make all physical measurements (clock, camera, mount encoders) near same time
                timestamp = time.time()
                frame = camera.get_frame()
                mount_position = mount.get_position()

                try:
                    sc_eq = track.plate_solve(
                        frame,
                        camera_width=camera.field_of_view[1]
                    )
                    sc_eq.obstime = Time(timestamp, format='unix')
                    sc_eq.location = location
                    sc_topo = sc_eq.transform_to('altaz')
                    print('Solution found!')
                    observations.append({
                        'unix_timestamp': timestamp,
                        'encoder_0': mount_position.encoder_0.deg,
                        'encoder_1': mount_position.encoder_1.deg,
                        'sky_az': sc_topo.az.deg,
                        'sky_alt': sc_topo.alt.deg,
                        'sky_ra': sc_eq.ra.deg,
                        'sky_dec': sc_eq.dec.deg,
                    })
                    hdu = fits.PrimaryHDU(frame)
                    try:
                        hdu.writeto(os.path.join(
                            observations_dir,
                            f'camera_frame_{num_solutions:03d}.fits'
                        ))
                    except OSError as e:
                        print('Trouble saving image: ' + str(e))
                    num_solutions += 1
                    break
                except track.NoSolutionException:
                    print('No solution.')

        print('Plate solver found solutions at {} of {} positions.'.format(
            num_solutions,
            len(positions)
        ))

        if num_solutions == 0:
            print("Can't solve mount model without any usable observations. Aborting.")
            sys.exit(1)
        elif num_solutions < args.min_positions:
            if not click.confirm(f'WARNING: You asked for at least {args.min_positions} positions '
                                 f'but only {num_solutions} can be used. Pointing accuracy may be '
                                 f'affected. Continue to solve for model parameters anyway?',
                                 default=True):
                sys.exit(1)

        observations = pd.DataFrame(observations)

        observations_filename = os.path.join(
            observations_dir,
            'observations_dataframe.pickle'
        )
        print(f'Saving observations to {observations_filename}')
        with open(observations_filename, 'wb') as f:
            pickle.dump(observations, f, pickle.HIGHEST_PROTOCOL)

        try:
            print('Solving for mount model parameters...', end='', flush=True)
            model_params, result = track.model.solve_model(observations)
            print('done.')
            print(result)

            rms_error = np.sqrt(2 * result.cost / len(observations))
            if rms_error > args.max_rms_error:
                if not click.confirm(f'WARNING: RMS error {rms_error:.4f} > '
                                     f'{args.max_rms_error:.4f} degrees, save this solution '
                                     f'anyway?', default=True):
                    sys.exit(1)
            else:
                print(f'RMS error: {rms_error:.4f} degrees')

            model_param_set = ModelParamSet(
                model_params=model_params,
                guide_cam_orientation=Longitude(args.guide_cam_orientation*u.deg),
                location=location,
                timestamp=time.time(),
            )

            params_filename = os.path.join(
                observations_dir,
                'model_params.pickle'
            )
            print(f'Saving model parameters to {params_filename}')
            with open(params_filename, 'wb') as f:
                pickle.dump(model_param_set, f, pickle.HIGHEST_PROTOCOL)

            print('Making this set of model parameters the default')
            track.model.save_default_param_set(model_param_set)

        except track.model.NoSolutionException as e:
            print('failed: {}'.format(str(e)))

    except RuntimeError as e:
        print(str(e))
        print('Alignment was not completed.')
    except KeyboardInterrupt:
        print('Got CTRL-C, shutting down...')
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...', end='', flush=True)
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

        if args.telem_enable:
            telem_logger.stop()

if __name__ == "__main__":
    main()
