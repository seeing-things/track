#!/usr/bin/env python

import config
import configargparse
import ephem
import datetime
import math

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES, formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
parser.add_argument('--timestamp', required=False, help='UNIX timestamp', type=float)

subparsers = parser.add_subparsers(title='modes', dest='mode')

parser_star = subparsers.add_parser('star', help='named star mode')
parser_star.add_argument('name', help='name of star')

parser_star = subparsers.add_parser('solarsystem', help='named solar system body mode')
parser_star.add_argument('name', help='name of planet or moon')

args = parser.parse_args()

# Create a PyEphem Observer object
observer = ephem.Observer()
observer.lat = args.lat
observer.lon = args.lon
observer.elevation = args.elevation

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

if args.timestamp is not None:
    observer.date = ephem.Date(datetime.datetime.utcfromtimestamp(args.timestamp))
else:
    observer.date = ephem.Date(datetime.datetime.utcnow())

target.compute(observer)

position = {
    'az': target.az * 180.0 / math.pi,
    'alt': target.alt * 180.0 / math.pi,
}

print('Expected position: ' + str(position))
