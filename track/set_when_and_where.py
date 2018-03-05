#!/usr/bin/env python

"""Sets mount time and location. Only supports NexStar mounts."""

import math
import time
import point
import ephem
import track

def main():

    parser = track.ArgParser()
    parser.add_argument('--mount-path', help='serial device node for connection to telescope', default='/dev/ttyUSB0')
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
    mount = point.NexStar(args.mount_path)
    mount.set_location(lat, lon)
    mount.set_time(time.time())

if __name__ == "__main__":
    main()
