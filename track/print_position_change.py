#!/usr/bin/env python

import track

def main():

    parser = track.ArgParser()
    parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
    args = parser.parse_args()

    # Create object with base type TelescopeMount
    mount = track.NexStarMount(args.scope)

    position_start = mount.get_azalt()

    while True:
        position = mount.get_azalt()
        position_change = {
            'az': track.wrap_error(position['az'] - position_start['az']) * 3600.0,
            'alt': track.wrap_error(position['alt'] - position_start['alt']) * 3600.0,
        }
        print(str(position_change))

if __name__ == "__main__":
    main()
