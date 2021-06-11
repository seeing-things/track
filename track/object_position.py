#!/usr/bin/env python3

"""Prints the position of an object from observer's location."""

import sys
import datetime
import math
import ephem
import ephem.stars
import track


def main():
    """See module docstring at the top of this file."""

    parser = track.ArgParser()

    parser.add_argument(
        '--lat',
        required=True,
        help='latitude of observer (+N)')
    parser.add_argument(
        '--lon',
        required=True,
        help='longitude of observer (+E)')
    parser.add_argument(
        '--elevation',
        required=True,
        help='elevation of observer (m)',
        type=float)
    parser.add_argument(
        '--timestamp',
        required=False,
        help='UNIX timestamp',
        type=float)

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
        print('In named star mode: looking up \'{}\''.format(args.name))
        target = None
        for name, _ in ephem.stars.stars.items():
            if args.name.lower() == name.lower():
                print('Found named star: \'{}\''.format(name))
                target = ephem.star(name)
                break
        if target is None:
            raise Exception('The named star \'{}\' isn\' present in PyEphem.'.format(args.name))

    # Get the PyEphem Body object corresonding to the given named solar system body
    elif args.mode == 'solarsystem':
        print('In named solar system body mode: looking up \'{}\''.format(args.name))
        # pylint: disable=protected-access
        ss_objs = [name.lower() for _, _, name in ephem._libastro.builtin_planets()]
        if args.name.lower() in ss_objs:
            body_type = None
            for attr in dir(ephem):
                if args.name.lower() == attr.lower():
                    body_type = getattr(ephem, attr)
                    print('Found solar system body: \'{}\''.format(attr))
                    break
            assert body_type is not None
            target = body_type()
        else:
            raise Exception(
                'The solar system body \'{}\' isn\'t present in PyEphem.'.format(args.name)
            )
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
