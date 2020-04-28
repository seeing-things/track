#!/usr/bin/env python3

"""Blind tracking of objects.

This program uses a feedback control loop to cause a telescope mount to track an object based on
the object's predicted position in the sky and the mount's self-reported encoder positions. It is
called "blind" tracking because it does not rely on any direct measurement or observation of the
object to function; it relies entirely on the accuracy of the mount alignment model and the
accuracy of the predicted position.

The object may be anything supported by the PyEmphem package which includes built-in lists of
bright stars and solar system objects, ephemeris derived from two-line element (TLE) file for
Earth-orbiting artificial satellites, and raw RA/DEC coordinates.

An optional gamepad can be used to provide adjustments to the pointing in real time.
These adjustments are relative to the motion vector of the object across the sky.
"""

from configargparse import Namespace
import os
import sys
import ephem
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
import click
import track
from track.control import Tracker
from track.gamepad import Gamepad
from track.laser import LaserPointer
from track.mounts import MeridianSide
from track.errorsources import BlindErrorSource
from track.targets import PyEphemTarget
from track.telem import TelemLogger


def make_target(args: Namespace) -> PyEphemTarget:
    """Create a PyEphem object to use as a target"""

    # Create a PyEphem Body object corresonding to the TLE file
    if args.mode == 'tle':
        print('In TLE file mode: \'{}\'.'.format(args.file))
        tle = []
        with open(args.file) as tlefile:
            for line in tlefile:
                tle.append(line)
        target = ephem.readtle(tle[0], tle[1], tle[2])

    # Create a PyEphem Body object corresonding to the given fixed coordinates
    elif args.mode == 'coord':
        print('In fixed-body coordinate mode: (RA {}, dec {}).'.format(args.ra, args.dec))
        target = ephem.FixedBody(_ra=np.radians(args.ra), _dec=np.radians(args.dec))

    # Get the PyEphem Body object corresonding to the given named star
    elif args.mode == 'star':
        print('In named star mode: \'{}\''.format(args.name))
        target = ephem.star(args.name)

    # Get the PyEphem Body object corresonding to the given named solar system body
    elif args.mode == 'solarsystem':
        print('In named solar system body mode: \'{}\''.format(args.name))
        ss_objs = [name for _, _, name in ephem._libastro.builtin_planets()]
        if args.name in ss_objs:
            body_type = getattr(ephem, args.name)
            target = body_type()
        else:
            raise Exception(
                'The solar system body \'{}\' isn\'t present in PyEphem.'.format(args.name)
            )
    else:
        print('No target specified!')
        sys.exit(1)

    return target


def main():
    """See module docstring"""

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
        help='side of meridian for equatorial mounts to prefer',
        default=MeridianSide.WEST.name.lower(),
        choices=tuple(m.name.lower() for m in MeridianSide),
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
        '--laser-ftdi-serial',
        help='serial number of laser pointer FTDI device',
    )

    subparsers = parser.add_subparsers(title='modes', dest='mode')

    parser_tle = subparsers.add_parser('tle', help='TLE file mode')
    parser_tle.add_argument('file', help='filename of two-line element (TLE) target ephemeris')

    parser_coord = subparsers.add_parser('coord', help='fixed-body coordinate mode')
    parser_coord.add_argument('ra', help='fixed body\'s right ascension', type=float)
    parser_coord.add_argument('dec', help='fixed body\'s declination', type=float)

    parser_star = subparsers.add_parser('star', help='named star mode')
    parser_star.add_argument('name', help='name of star')

    parser_star = subparsers.add_parser('solarsystem', help='named solar system body mode')
    parser_star.add_argument('name', help='name of planet or moon')

    args = parser.parse_args()

    # Set priority of this thread to realtime. Do this before constructing objects since priority
    # is inherited and some critical threads are created by libraries we have no direct control
    # over.
    os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(11))

    # Create object with base type TelescopeMount
    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path)
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)


    def target_offset_callback(tracker: Tracker) -> False:
        """Use gamepad integrator to adjust the target predicted position

        This is registered as a callback for the Tracker object. An offset is only applied when the
        'B' button on the gamepad is held down to prevent unintentional disruption of the control
        system action.

        Args:
            tracker: The instance of Tracker that called this function.

        Returns:
            False to indicate that the control loop should continue execution as normal.
        """
        #pylint: disable=unused-argument
        if game_pad.state.get('BTN_EAST', 0) == 1:
            x, y = game_pad.get_integrator()
            gamepad_polar = x + 1j*y
            error_source.target_position_offset = BlindErrorSource.PositionOffset(
                direction=Angle(np.angle(gamepad_polar)*u.rad),
                separation=Angle(np.abs(gamepad_polar)*u.deg),
            )
        else:
            error_source.target_position_offset = None
        return False

    # Load a MountModel object
    try:
        mount_model = track.model.load_stored_model()
    except track.model.StaleParametersException:
        if click.confirm('Stored model parameters are stale. Use anyway?', default=False):
            mount_model = track.model.load_stored_model(max_age=None)
        else:
            print('Aborting. To refresh model parameters, run align program.')
            sys.exit(1)

    # Create object with base type ErrorSource
    error_source = track.BlindErrorSource(
        mount=mount,
        mount_model=mount_model,
        target=PyEphemTarget(make_target(args), mount_model.location),
        meridian_side=MeridianSide[args.meridian_side.upper()]
    )
    telem_sources = {'error_blind': error_source}

    tracker = Tracker(
        mount=mount,
        error_source=error_source,
    )
    telem_sources['tracker'] = tracker

    try:
        laser = LaserPointer(serial_num=args.laser_ftdi_serial)
    except OSError:
        print('Could not connect to laser pointer FTDI device.')
        laser = None

    try:
        # Create gamepad object and register callback
        game_pad = Gamepad(
            left_gain=2.0,  # left stick degrees per second
            right_gain=0.5, # right stick degrees per second
            int_limit=5.0,  # max correction in degrees for either axis
        )
        game_pad.integrator_mode = True
        if laser is not None:
            game_pad.register_callback('BTN_SOUTH', laser.set)
        telem_sources['gamepad'] = game_pad
        tracker.register_callback(target_offset_callback)
        print('Gamepad found and registered.')
    except RuntimeError:
        print('No gamepads found.')

    if args.telem_enable:
        telem_logger = TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources=telem_sources,
        )
        telem_logger.start()

    try:
        tracker.run()
    except KeyboardInterrupt:
        print('Got CTRL-C, shutting down...')
    except Exception as e:
        print('Unhandled exception: ' + str(e))
        import traceback
        traceback.print_exc()
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...', end='', flush=True)
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

        if args.telem_enable:
            telem_logger.stop()

        try:
            game_pad.stop()
        except:
            pass

if __name__ == "__main__":
    main()
