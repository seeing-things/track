#!/usr/bin/env python3

"""Perform automated alignment of a telescope mount.

This program automates the somewhat tedious process of aligning the mount. It will do the following
things:
1) Generate a list of positions on the sky to point at
2) For each position:
   a) Point the mount at the position
   b) Capture an image with a camera
   c) Use the astrometry.net plate solver to determine the sky coordinates of the image
   d) Store a timestamp and the mount's encoder positions
3) Use the set of observations to solve for mount model parameters
4) Store the mount model parameters on disk for future use during the same observing session
"""

import sys
import time
from typing import List, Optional, NamedTuple
import pandas as pd
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle
import track
from track import cameras
from track.model import ModelParameters, ModelParamSet, MountModel
from track.mounts import MeridianSide
from track.targets import FixedTopocentricTarget


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
    parser.add_argument(
        '--meridian-side',
        help='do alignment using only positions on this side of meridian defined by mount pole',
        default=None,
        choices=tuple(m.name.lower() for m in MeridianSide),
    )
    parser.add_argument(
        '--lat',
        required=True,
        help='latitude of observer (+N)',
        type=float,
    )
    parser.add_argument(
        '--lon',
        required=True,
        help='longitude of observer (+E)',
        type=float,
    )
    parser.add_argument(
        '--elevation',
        required=True,
        help='elevation of observer (m)',
        type=float
    )
    parser.add_argument(
        '--mount-pole-alt',
        required=True,
        help='altitude of mount pole above horizon (deg)',
        type=float
    )
    parser.add_argument(
        '--loop-bw',
        help='control loop bandwidth (Hz)',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--loop-damping',
        help='control loop damping factor',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--telem-enable',
        help='enable logging of telemetry to database',
        action='store_true'
    )
    parser.add_argument(
        '--telem-db-host',
        help='hostname of InfluxDB database server',
        default='localhost'
    )
    parser.add_argument(
        '--telem-db-port',
        help='port number of InfluxDB database server',
        default=8086,
        type=int
    )
    parser.add_argument(
        '--telem-period',
        help='telemetry sampling period in seconds',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--min-positions',
        help='minimum number of positions to add to mount alignment model',
        default=10,
        type=int
    )
    parser.add_argument(
        '--timeout',
        help='max time to wait for mount to converge on a new position',
        default=120.0,
        type=float
    )
    parser.add_argument(
        '--max-tries',
        help='max number of plate solving attempts at each position',
        default=3,
        type=int
    )
    parser.add_argument(
        '--min-alt',
        help='minimum altitude of alignment positions in degrees',
        default=20.0,
        type=float
    )
    cameras.add_program_arguments(parser)
    args = parser.parse_args()

    if args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    elif args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    location = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.elevation*u.m)

    # Parameters for use during alignment. This is meant to be just good enough that the model is
    # able to tell roughly which direction is up so that the positions used during alignment are
    # all above the horizon.
    model_params = ModelParameters(
        axis_0_offset=Angle(0*u.deg),
        axis_1_offset=Angle(0*u.deg),
        pole_rot_axis_az=Angle(90*u.deg),
        pole_rot_angle=Angle((90.0 - args.mount_pole_alt)*u.deg),
    )

    # target and meridian_side will be populated later
    error_source = track.BlindErrorSource(
        mount=mount,
        mount_model=MountModel(model_params),
        target=None,
        meridian_side=None
    )
    telem_sources = {'error_blind': error_source}

    tracker = track.Tracker(
        mount=mount,
        error_source=error_source,
        loop_bandwidth=args.loop_bw,
        damping_factor=args.loop_damping
    )
    telem_sources['tracker'] = tracker
    tracker.stop_on_timer = True
    tracker.max_run_time = args.timeout
    tracker.stop_when_converged = True
    tracker.converge_max_error_mag = Angle(2.0 * u.deg)

    camera = cameras.make_camera_from_args(args)

    if args.telem_enable:
        telem_logger = track.TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources=telem_sources,
        )
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

    # pylint: disable=broad-except
    try:
        observations = []
        num_solutions = 0
        for idx, position in enumerate(positions):

            print(f'Moving to position {idx} of {len(positions)}: {str(position.position)}')

            error_source.target = FixedTopocentricTarget(position.position)
            error_source.meridian_side = position.meridian_side
            stop_reason = tracker.run()
            mount.safe()
            if stop_reason != 'converged':
                raise RuntimeError('Unexpected tracker stop reason: "{}"'.format(stop_reason))

            print('Converged on the target position. Attempting plate solving.')

            # plate solver doesn't always work on the first try
            for i in range(args.max_tries):

                print('\tPlate solver attempt {} of {}...'.format(i + 1, args.max_tries), end='')

                timestamp = time.time()
                frame = camera.get_frame()

                try:
                    sc_eq = track.plate_solve(
                        frame,
                        camera_width=camera.field_of_view[1]
                    )
                    sc_eq.obstime = Time(timestamp, format='unix')
                    sc_eq.location = location
                    sc_topo = sc_eq.transform_to('altaz')
                    print('Solution found!')
                    mount_position = mount.get_position()
                    observations.append({
                        'unix_timestamp': timestamp,
                        'encoder_0': mount_position.encoder_0.deg,
                        'encoder_1': mount_position.encoder_1.deg,
                        'sky_az': sc_topo.az.deg,
                        'sky_alt': sc_topo.alt.deg,
                    })
                    num_solutions += 1
                    break
                except track.NoSolutionException:
                    print('No solution.')

        print('Plate solver found solutions at {} of {} positions.'.format(
            num_solutions,
            len(positions)
        ))
        observations = pd.DataFrame(observations)

        try:
            print('Solving for mount model parameters...', end='')
            model_params, result = track.model.solve_model(observations)
            model_param_set = ModelParamSet(
                model_params=model_params,
                location=location,
                timestamp=time.time(),
            )
            print('success!')
            print(result)
            filename = track.model.DEFAULT_MODEL_FILENAME
            print('Saving model parameters to {}'.format(filename))
            track.model.save_default_param_set(model_param_set)
        except track.model.NoSolutionException as e:
            print('failed: {}'.format(str(e)))

    except RuntimeError as e:
        print(str(e))
        print('Alignment was not completed.')
    except KeyboardInterrupt:
        print('Got CTRL-C, shutting down...')
    except Exception as e:
        print('Unhandled exception: ' + str(e))
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...')
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

    if args.telem_enable:
        telem_logger.stop()

if __name__ == "__main__":
    main()
