#!/usr/bin/env python

import config
import configargparse
import track
import mounts
import errorsources
import ephem

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES, formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.5, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=0.5, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.25, type=float)
parser.add_argument('--backlash-az', help='backlash in azimuth (arcseconds)', default=0.0, type=float)
parser.add_argument('--backlash-alt', help='backlash in altitude (arcseconds)', default=0.0, type=float)
parser.add_argument('--align-dir-az', help='azimuth alignment approach direction (-1 or +1)', default=+1, type=int)
parser.add_argument('--align-dir-alt', help='altitude alignment approach direction (-1 or +1)', default=+1, type=int)

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

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)
mount.set_backlash('az', args.align_dir_az, args.backlash_az / 3600.0)
mount.set_backlash('alt', args.align_dir_alt, args.backlash_alt  / 3600.0)

# Create a PyEphem Observer object
observer = ephem.Observer()
observer.lat = args.lat
observer.lon = args.lon
observer.elevation = args.elevation

# Create a PyEphem Body object corresonding to the TLE file
if args.mode == 'tle':
    print('In TLE file mode: \'{}\'.'.format(args.file))
    tle = []
    with open(args.file) as tlefile:
        for line in tlefile:
            tle.append(line)
    target = ephem.readtle(tle[0], tle[1], tle[2])

# Create a PyEphem Body object corresonding to the given fixed coordinates
if args.mode == 'coord':
    print('In fixed-body coordinate mode: (RA {}, dec {}).'.format(args.ra, args.dec))
    target = ephem.FixedBody(_ra=args.ra, _dec=args.dec)

# Get the PyEphem Body object corresonding to the given named star
if args.mode == 'star':
    print('In named star mode: \'{}\''.format(args.name))
    target = ephem.star(args.name)

# Get the PyEphem Body object corresonding to the given named solar system body
if args.mode == 'solarsystem':
    print('In named solar system body mode: \'{}\''.format(args.name))
    ss_objs = [name for _0, _1, name in ephem._libastro.builtin_planets()]
    if args.name in ss_objs:
        body_type = getattr(ephem, args.name)
        target = body_type()
    else:
        raise Exception('The solar system body \'{}\' isn\'t present in PyEphem.'.format(args.name))

# Create object with base type ErrorSource
error_source = errorsources.BlindErrorSource(mount, observer, target)

tracker = track.Tracker(
    mount = mount, 
    error_source = error_source, 
    update_period = args.loop_period,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

try:
    tracker.run()
except KeyboardInterrupt:
    print('Goodbye!')
    pass
