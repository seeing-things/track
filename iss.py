#!/usr/bin/env python

import config
import ephem
import time
import urllib2

parser = config.ArgParser()
parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
parser.add_argument('--elevation', required=False, default=0.0, help='elevation of observer (m)', type=float)
args = parser.parse_args()

# Grab the latest space station TLE file from Celestrak
tlefile = urllib2.urlopen('http://celestrak.com/NORAD/elements/stations.txt')
tle = []
for line in tlefile:
	tle.append(line)
tlefile.close()

# Top entry in the station.txt file should be for ISS
iss = ephem.readtle(tle[0], tle[1], tle[2])

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
