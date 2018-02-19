#!/usr/bin/env python

import track

def main():

    parser = track.ArgParser()
    parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
    args = parser.parse_args()

    # Create object with base type TelescopeMount
    mount = track.NexStarMount(args.scope)

    position_start = mount.get_position()

    while True:
        position = mount.get_position()
        position_change = {}
        for axis in mount.get_axis_names():
            position_change[axis] = track.wrap_error(position[axis] - position_start[axis]) * 3600.0
        print(str(position_change))

if __name__ == "__main__":
    main()
