#!/usr/bin/env python3

"""Prints the position of an object from observer's location."""

import sys
import datetime
import math
import ephem
import ephem.stars
from track.config import ArgParser
from track import gps_client


def main():
    """See module docstring at the top of this file."""
    parser = ArgParser()
    parser.add_argument('--timestamp', required=False, help='UNIX timestamp', type=float)

    subparsers = parser.add_subparsers(title='modes', dest='mode')

    parser_star = subparsers.add_parser('star', help='named star mode')
    parser_star.add_argument(
        'name',
        help='name of star',
        type=str.title,
        choices=sorted(ephem.stars.stars.keys()),
    )

    parser_star = subparsers.add_parser('solarsystem', help='named solar system body mode')
    parser_star.add_argument(
        'name',
        help='name of planet or moon',
        type=str.capitalize,
        # pylint: disable=protected-access
        choices=[planet[2] for planet in ephem._libastro.builtin_planets()],
    )

    gps_client.add_program_arguments(parser)
    args = parser.parse_args()

    # Create a PyEphem Observer object
    location = gps_client.make_location_from_args(args)
    observer = ephem.Observer()
    observer.lat = location.lat.deg
    observer.lon = location.lon.deg
    observer.elevation = location.height.value

    # Get the PyEphem Body object corresponding to the given named star
    if args.mode == 'star':
        print(f"In named star mode: looking up '{args.name}'")
        target = ephem.star(args.name)

    # Get the PyEphem Body object corresponding to the given named solar system body
    elif args.mode == 'solarsystem':
        print(f'In named solar system body mode: {args.name}')
        body_type = getattr(ephem, args.name)
        target = body_type()
    else:
        print('You must specify a target.')
        sys.exit(1)

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


if __name__ == "__main__":
    main()
