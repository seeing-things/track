#!/usr/bin/env python

import ephem
import time
import requests
import track

def main():

    parser = track.ArgParser()
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument('--elevation', required=False, default=0.0, help='elevation of observer (m)', type=float)
    args = parser.parse_args()

    # Grab the latest space station TLE file from Celestrak
    stations = requests.get('http://celestrak.com/NORAD/elements/stations.txt').text.splitlines()

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
            print('Az: %s Alt: %s' % (iss.az, iss.alt))
            time.sleep(1)

if __name__ == "__main__":
    main()
