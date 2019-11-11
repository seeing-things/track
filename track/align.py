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
import numpy as np
import pandas as pd
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer
import cv2
import asi
import track


def alt_from_ha_dec(ha, dec, mount_pole_altitude):
    """Get the approximate altitude of a target specified with a local hour angle and declination.

    Args:
        ha: Target local hour angle as an astropy Angle object.
        dec: Target declination as an astropy Angle object.
        mount_pole_altitude: Altitude of the mount's pole in degrees above the horizon.

    Returns:
        The altitude of the target as an astropy Angle object.
    """

    # This is a fake location required in order to determine the altitude of each HEALPix pixel.
    # In these conversions, which only need to be approximate, the longitude and elevation of the
    # observer don't matter. The "latitude" is set to the altitude of the mount's pole since our
    # HEALPix hour angles and declinations are relative to the mount's pole location and not the
    # celestial pole.
    location = EarthLocation(lat=mount_pole_altitude*u.deg, lon=0*u.deg, height=0*u.m)

    # The time should be relatively unimportant for these calculations; it is only necessary
    # because astropy does not provide a direct transformation between local coordinates given as
    # hour angle / declination and local coordinates in azimuth / altitude format. Therefore, we
    # must transform first to equatorial coordinates (right ascension / declination) before
    # converting finally to azimuth / altitude.
    t = Time(time.time(), format='unix', location=location)
    st = t.sidereal_time('mean')
    ra = st - ha
    me = Observer(location=location)
    sc = SkyCoord(ra, dec, frame='icrs')
    return me.altaz(t, target=sc).alt

def generate_positions(min_positions, mount_pole_altitude, min_altitude=0.0, meridian_side=None):
    """Generate a list of equally spaced positions on one hemisphere of the sky to search.

    The list of positions generated will be a subset of pixels generated by the HEALPix algorithm,
    which is a method of pixelization of a sphere where each "pixel" has equal area. In the
    context of mount alignment we not so much concerned with the equal area property, but we do
    want positions that are roughly evenly distributed over the sphere, and HEALPix does a
    reasonable job of this as well. The full set of HEALPix pixels is filtered to exclude pixels
    that are below a minimum altitude threshold and (optionally) those that are on the opposite
    side of the meridian. HEALPix is oriented with the top co-located with the pole of the mount.

    Args:
        min_positions: Minimum number of positions. The actual number of positions returned may be
            larger than requested.
        mount_pole_altitude: Altitude of the mount's pole in degrees. For a German equatorial
            mount in the Northern hemisphere this is usually equal to the current latitude, however
            this is not required.
        min_altitude: Restrict positions to be above this altitude in degrees.
        meridian_side: A string, 'east' or 'west', or None. If specified, restricts positions to
            only be on this side of the meridian.

    Returns:
        A list of dicts with keys 'ha' and 'dec' giving hour angles and declinations in degrees.
            Since this is used for alignment and no assumptions are made about the orientation
            of the mount, particularly the location of the mount's pole, these coordinates are
            interpreted with respect to the mount's startup position. In other words, declination
            0 means the optical tube is pointed towards the mount's physical pole, and not
            necessarily towards Polaris. Similarly, hour angle 0 means that the counter weight is
            pointed down, and this does not necessarily correspond to the great circle passing
            through the local zenith and the celestial pole.
    """
    level = 0
    while True:
        healpix = HEALPix(nside=2**level)
        positions = []
        for i in range(healpix.npix):

            # interpret each HEALPix as an hour angle and declination coordinate
            (ha, dec) = healpix.healpix_to_lonlat(i)

            # skip points on wrong side of meridian
            if ha.deg <= 180.0 and meridian_side == 'east':
                continue

            if ha.deg > 180.0 and meridian_side == 'west':
                continue

            # skip points below min altitude threshold
            alt = alt_from_ha_dec(ha, dec, mount_pole_altitude)
            if alt.deg < min_altitude:
                continue

            positions.append({'ha': ha.deg, 'dec': dec.deg})

        if len(positions) >= min_positions:
            break

        # not enough positions -- try again with a higher level HEALPix
        level += 1

    return positions


def camera_setup(gain, exposure_time, binning):
    """Initialize and configure ZWO ASI camera.

    TODO: Abstract this ZWO-specific code behind some generic interface.

    Args:
        gain (int): Camera gain setting.
        exposure_time (float): Exposure time in seconds. Will be rounded down to the nearest
            microsecond.
        binning (int): Camera binning.

    Returns:
        A tuple containing the following:
            info: An ASI_CAMERA_INFO object.
            width: Width of the frame in pixels.
            height: Height of the frame in pixels.
            frame_size: Frame size in bytes. The camera is configured for 16-bit mode, so there
                are two bytes per pixel.

    Raises:
        RuntimeError for any camera related problems.
    """
    if asi.ASIGetNumOfConnectedCameras() == 0:
        raise RuntimeError('No cameras connected')
    info = asi.ASICheck(asi.ASIGetCameraProperty(0))
    width = info.MaxWidth // binning
    height = info.MaxHeight // binning
    frame_size = width * height * 2
    asi.ASICheck(asi.ASIOpenCamera(info.CameraID))
    asi.ASICheck(asi.ASIInitCamera(info.CameraID))
    asi.ASICheck(asi.ASISetROIFormat(
        info.CameraID,
        width,
        height,
        binning,
        asi.ASI_IMG_RAW16
    ))
    asi.ASICheck(asi.ASISetControlValue(
        info.CameraID,
        asi.ASI_EXPOSURE,
        int(exposure_time * 1e6),
        asi.ASI_FALSE
    ))
    asi.ASICheck(asi.ASISetControlValue(info.CameraID, asi.ASI_GAIN, gain, asi.ASI_FALSE))
    asi.ASICheck(asi.ASISetControlValue(info.CameraID, asi.ASI_MONO_BIN, 1, asi.ASI_FALSE))

    return info, width, height, frame_size


def camera_take_exposure(info, width, height, frame_size):
    """Take an exposure with the camera.

    Args:
        info: An ASI_CAMERA_INFO object.
        width: Width of the frame in pixels.
        height: Height of the frame in pixels.
        frame_size: Size of the frame in bytes (not necessarily equal to width*height).

    Returns:
        A numpy array containing a debayered grayscale camera frame.

    Raises:
        RuntimeError if the exposure failed.
    """
    asi.ASICheck(asi.ASIStartExposure(info.CameraID, asi.ASI_FALSE))
    while True:
        time.sleep(0.01)
        status = asi.ASICheck(asi.ASIGetExpStatus(info.CameraID))
        if status == asi.ASI_EXP_SUCCESS:
            break
        if status == asi.ASI_EXP_FAILED:
            raise RuntimeError('Exposure failed')
    frame = asi.ASICheck(asi.ASIGetDataAfterExp(info.CameraID, frame_size))
    frame = frame.view(dtype=np.uint16)
    frame = np.reshape(frame, (height, width))
    return cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)


def main():
    """Run the alignment procedure! See module docstring for a description."""

    parser = track.ArgParser()
    parser.add_argument(
        '--camera-res',
        help='guidescope camera resolution in arcseconds per pixel',
        required=True,
        type=float
    )
    parser.add_argument(
        '--exposure-time',
        help='camera exposure time in seconds',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--gain',
        help='camera gain',
        default=400,
        type=int
    )
    parser.add_argument(
        '--binning',
        help='camera binning',
        default=4,
        type=int
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
    parser.add_argument(
        '--laser-ftdi-serial',
        help='serial number of laser pointer FTDI device',
    )
    args = parser.parse_args()

    # This program only supports Gemini mounts
    # TODO: Abstract Gemini-specific code
    if args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    # Create object with base type ErrorSource. No target for now; that will be populated later.
    error_source = track.BlindErrorSource(
        mount=mount,
        observer=None,
        target=None,
        meridian_side=args.meridian_side
    )
    telem_sources = {'error_blind': error_source}

    try:
        laser = track.LaserPointer(serial_num=args.laser_ftdi_serial)
    except OSError:
        print('Could not connect to laser pointer FTDI device.')
        laser = None

    try:
        # Create gamepad object and register callback
        game_pad = track.Gamepad(
            left_gain=2.0,  # left stick degrees per second
            right_gain=0.5,  # right stick degrees per second
            int_limit=5.0,  # max correction in degrees for either axis
        )
        game_pad.integrator_mode = True
        if laser is not None:
            game_pad.register_callback('BTN_SOUTH', laser.set)
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
    tracker.converge_max_error_mag = 2.0

    camera_info, frame_width, frame_height, frame_size = camera_setup(
        gain=args.gain,
        exposure_time=args.exposure_time,
        binning=args.binning
    )

    if args.telem_enable:
        telem_logger = track.TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources=telem_sources,
        )
        telem_logger.start()

    positions = generate_positions(
        min_positions=args.min_positions,
        mount_pole_altitude=args.mount_pole_alt,
        min_altitude=args.min_alt,
        meridian_side=args.meridian_side
    )

    # pylint: disable=broad-except
    try:
        data = pd.DataFrame(columns=['time', 'encoder_ra', 'encoder_dec', 'sky_ra', 'sky_dec'])
        num_solutions = 0
        for idx, position in enumerate(positions):

            print('Moving to position {} of {}: {}'.format(idx, len(positions), str(position)))

            error_source.target = position
            stop_reason = tracker.run()
            mount.safe()
            if stop_reason != 'converged':
                raise RuntimeError('Unexpected tracker stop reason: "{}"'.format(stop_reason))

            print('Converged on the target position. Attempting plate solving.')

            # plate solver doesn't always work on the first try
            for i in range(args.max_tries):

                print('\tPlate solver attempt {} of {}...'.format(i + 1, args.max_tries), end='')

                timestamp = time.time()
                frame = camera_take_exposure(camera_info, frame_width, frame_height, frame_size)

                try:
                    sc = track.plate_solve(
                        frame,
                        camera_width=(camera_info.MaxWidth * args.camera_res / 3600.0)
                    )
                    print('Solution found!')
                    mount_position = mount.get_position()
                    data.append({
                        'unix_time': timestamp,
                        'encoder_ra': mount_position['pra'],
                        'encoder_dec': mount_position['pdec'],
                        'sky_ra': sc.ra.deg,
                        'sky_dec': sc.dec.deg,
                    })
                    num_solutions += 1
                    break
                except track.NoSolutionException:
                    print('No solution.')

        print('Plate solver found solutions at {} of {} positions.'.format(
            num_solutions,
            len(positions)
        ))

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

    try:
        game_pad.stop()
    # pylint: disable=bare-except
    except:
        pass

if __name__ == "__main__":
    main()
