#!/usr/bin/env python

import track
import mounts
import errorsources
import ephem
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('tle', help='filename of two-line element (TLE) target ephemeris')
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.5, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=0.5, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.25, type=float)
parser.add_argument('--backlash-az', help='backlash in azimuth (deg)', default=0.0, type=float)
parser.add_argument('--backlash-alt', help='backlash in altitude (deg)', default=0.0, type=float)
parser.add_argument('--align-dir-az', help='azimuth alignment approach direction (-1 or +1)', default=+1, type=int)
parser.add_argument('--align-dir-alt', help='altitude alignment approach direction (-1 or +1)', default=+1, type=int)
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)
mount.set_backlash('az', args.align_dir_az, args.backlash_az)
mount.set_backlash('alt', args.align_dir_alt, args.backlash_alt)

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
