#!/usr/bin/env python

import config
import math
import time
import nexstar
import ephem

parser = config.ArgParser()
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
args = parser.parse_args()

# We want to parse the latitude and longitude exactly the same way as our other
# scripts do: by letting ephem.Angle do the parsing itself. But they explicitly
# disallow us from creating our own Angle objects directly, so we're forced to
# work around that by using an object (Observer) that has Angles in it already.
observer = ephem.Observer()
observer.lat = args.lat
observer.lon = args.lon

# Convert to degrees
lat = observer.lat * 180.0 / math.pi
lon = observer.lon * 180.0 / math.pi

# Shove data into telescope
nexstar = nexstar.NexStar(args.scope)
nexstar.set_location(lat, lon)
nexstar.set_time(time.time())
