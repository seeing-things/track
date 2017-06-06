#!/usr/bin/env python

import track
import ephem
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tle', help='filename of two-line element (TLE) target ephemeris')
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = track.NexStarMount(args.scope)

# Create a PyEphem Observer object
observer = ephem.Observer()
observer.lat = args.lat
observer.lon = args.lon
observer.elevation = args.elevation

# Create a PyEphem Body object corresonding to the TLE file
tle = []
with open(args.tle) as tlefile:
    for line in tlefile:
        tle.append(line)
target = ephem.readtle(tle[0], tle[1], tle[2])

# Create object with base type ErrorSource
error_source = track.BlindErrorSource(mount, observer, target)

tracker = track.Tracker(
    mount = mount, 
    error_source = error_source, 
    update_period = 0.25,
    loop_bandwidth = 0.5,
    damping_factor = math.sqrt(2.0) / 2.0
)
tracker.start()

try:
    print('Press Enter to quit.')
    raw_input()
except KeyboardInterrupt:
    pass
finally:
    tracker.stop()
    print('Goodbye!')
