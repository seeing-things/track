#!/usr/bin/env python3

"""Prints position of ISS.

Simple program grabs the latest TLE for the ISS from Celestrak and then uses the PyEphem package
to print the current Az/Alt position of ISS relative to the observer's location.
"""

import time
import ephem
import requests
from track.config import ArgParser


def main():
    """See module docstring."""
    parser = ArgParser()
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument(
        '--elevation', required=False, default=0.0, help='elevation of observer (m)', type=float
    )
    args = parser.parse_args()

    # Grab the latest space station TLE file from Celestrak
    stations = requests.get(
        'http://celestrak.com/NORAD/elements/stations.txt', timeout=10
    ).text.splitlines()

    # Top entry in the station.txt file should be for ISS
    iss = ephem.readtle(str(stations[0]), str(stations[1]), str(stations[2]))

    # Create PyEphem Observer object with location information
    home = ephem.Observer()
    home.lon = args.lon
    home.lat = args.lat
    home.elevation = args.elevation

    # Print current Az-Alt of ISS once per second
    while True:
        home.date = ephem.now()
        iss.compute(home)
        print(f'Az: {iss.az} Alt: {iss.alt}')
        time.sleep(1)


if __name__ == "__main__":
    main()
