#!/usr/bin/env python3

"""Perform automated alignment with Gemini 2 mounts.

This program automates the somewhat tedious process of aligning the mount. For the selected
meridian side it will select a bright star, slew the mount to the vicinity of that star, use the
guidscope camera to center on that star, and send commands to Gemini 2 to add the star to the
alignment model (or to synchronize in the case of the first star). This process is repeated until
the desired number of stars have been added to the model or the program runs out of usable stars.
"""

import math
import time
import threading
import numpy as np
import ephem
import ephem.stars
import track


def pick_next_star(
        observer,
        meridian_side,
        min_alt=10.0,
        min_ra_separation=10.0,
        used_stars=[],
        rejected_stars=[]
    ):
    """Pick next star to use for alignment.

    Args:
        observer: An ephem.Observer object
        meridian_side: A string 'east' or 'west' indicating side of the meridian to pick from
        min_alt: Star altitude must be at least this in degrees
        min_ra_separation: Star must be separated from all previously used alignment stars by at
            least this many degrees in right ascension
        used_stars: List of names of stars already used for alignment
        rejected_stars: List of names of stars rejected for other reasons but which were not used
            for alignment. For example, a star that was previously picked by this function but
            which was obstructed by a building or clouds.
    """

    # Start with dict of all named stars PyEphem knows (copy so we don't modify the original).
    # Not sure if this is the best possible catalog of stars to use but it's the one readily at
    # hand. It may omit some bright stars that didn't happen to have proper names in the catalog
    # the PyEphem authors drew from.
    stars = ephem.stars.stars.copy()

    # filter the dict to reject stars that are not suitable
    for name in list(stars.keys()):

        # remove stars that were used or rejected previously
        if name in used_stars or name in rejected_stars:
            stars.pop(name, None)
            continue

        star = stars[name]
        star.compute(observer)

        # minimum altitude filter
        if star.alt * 180.0 / math.pi < min_alt:
            stars.pop(name, None)
            continue

        # meridian side filter
        if star.az < math.pi and meridian_side == 'west':
            stars.pop(name, None)
            continue
        elif star.az > math.pi and meridian_side == 'east':
            stars.pop(name, None)
            continue

        # reject stars that are too close in RA to stars already used for alignment
        for used_star_name in used_stars:
            used_star = ephem.star(used_star_name, observer)
            if abs(star.ra - used_star.ra) * 180.0 / math.pi < min_ra_separation:
                stars.pop(name, None)
                break

    if len(stars) == 0:
        raise RuntimeError('No named stars in PyEphem meet the selection criteria!')

    # return the brightest star remaining on the list
    return stars[min(stars, key=lambda name: stars[name].mag)]



def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--camera',
        help='device node path for tracking webcam',
        default='/dev/video0'
    )
    parser.add_argument(
        '--camera-res',
        help='webcam resolution in arcseconds per pixel',
        required=True,
        type=float
    )
    parser.add_argument(
        '--camera-bufs',
        help='number of webcam capture buffers',
        required=True,
        type=int
    )
    parser.add_argument(
        '--camera-exposure',
        help='webcam exposure level',
        default=3200,
        type=int
    )
    parser.add_argument(
        '--frame-dump-dir',
        help='directory to save webcam frames as jpeg files on disk',
    )
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
        help='side of meridian for equatorial mounts to prefer',
        default='west'
    )
    parser.add_argument(
        '--lat',
        required=True,
        help='latitude of observer (+N)'
    )
    parser.add_argument(
        '--lon',
        required=True,
        help='longitude of observer (+E)'
    )
    parser.add_argument(
        '--elevation',
        required=True,
        help='elevation of observer (m)',
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
        '--max-divergence',
        help='max divergence of optical and blind sources (degrees)',
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
        '--num-stars',
        help='number of stars to add to mount alignment model',
        default=5,
        type=int
    )
    parser.add_argument(
        '--timeout',
        help='if star is not found after this many seconds give up and move on to next star',
        default=120.0,
        type=float
    )
    parser.add_argument(
        '--min-alt',
        help='minimum altitude of alignment stars in degrees',
        default=20.0,
        type=float
    )
    parser.add_argument(
        '--min-ra-sep',
        help='minimum right ascension separation between all alignment stars in degrees',
        default=5.0,
        type=float
    )
    args = parser.parse_args()

    # This program only supports Gemini mounts
    if args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    # Create a PyEphem Observer object
    observer = ephem.Observer()
    observer.lat = args.lat
    observer.lon = args.lon
    observer.elevation = args.elevation

    # Create object with base type ErrorSource
    # No target for now; that will be populated later
    error_source = track.HybridErrorSource(
        mount=mount,
        observer=observer,
        target=None,
        cam_dev_path=args.camera,
        arcsecs_per_pixel=args.camera_res,
        cam_num_buffers=args.camera_bufs,
        cam_ctlval_exposure=args.camera_exposure,
        max_divergence=args.max_divergence,
        meridian_side=args.meridian_side,
        frame_dump_dir=args.frame_dump_dir,
    )
    telem_sources = {}
    telem_sources['error_hybrid'] = error_source
    telem_sources['error_blind'] = error_source.blind
    telem_sources['error_optical'] = error_source.optical

    try:
        # Create gamepad object and register callback
        game_pad = track.Gamepad(
            left_gain=2.0,  # left stick degrees per second
            right_gain=0.5,  # right stick degrees per second
            int_limit=5.0,  # max correction in degrees for either axis
        )
        game_pad.integrator_mode = True
        error_source.register_blind_offset_callback(game_pad.get_integrator)
        telem_sources['gamepad'] = game_pad
        print('Gamepad found and registered.')
    except RuntimeError:
        print('No gamepads found.')

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
    tracker.converge_error_state = 'optical'

    if args.telem_enable:
        telem_logger = track.TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources=telem_sources,
        )
        telem_logger.start()

    try:
        synced = False
        used_stars = []
        rejected_stars = []
        while True:

            # goto expected position of star
            target = pick_next_star(
                observer,
                meridian_side=args.meridian_side,
                min_alt=args.min_alt,
                min_ra_separation=args.min_ra_sep,
                used_stars=used_stars,
                rejected_stars=rejected_stars,
            )
            print('Slewing to ' + target.name)
            error_source.blind.target = target
            tracker.start_time = time.time()
            stop_reason = tracker.run()
            if stop_reason == 'converged':
                print('Converged on the target!')
                mount.mount.set_user_object_equatorial(
                    target.ra * 180.0 / math.pi,
                    target.dec * 180.0 / math.pi,
                    target.name
                )
                if not synced:
                    print('Synchronized to ' + target.name)
                    mount.mount.sync_to_object()
                    synced = True
                else:
                    print('Added ' + target.name + ' to mount model')
                    mount.mount.align_to_object()
                used_stars.append(target.name)
            elif stop_reason == 'timer expired':
                print('Timeout on ' + target.name + '; rejecting this star')
                rejected_stars.append(target.name)
            else:
                raise RuntimeError('Unexpected tracker stop reason: "' + stop_reason + '"')

            if len(used_stars) >= args.num_stars:
                break

        print('Alignment completed successfully!')

    except KeyboardInterrupt:
        print('Goodbye')

    if args.telem_enable:
        telem_logger.stop()

if __name__ == "__main__":
    main()