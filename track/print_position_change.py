#!/usr/bin/env python

import sys
import track

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--mount-type',
        help='select mount type (nexstar or gemini)',
        default='gemini'
    )
    parser.add_argument(
        '--mount-path',
        help='serial device node or hostname for mount command interface',
        default='/dev/ttyACM0'
    )
    args = parser.parse_args()

    # Create object with base type TelescopeMount
    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mout_path)
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    position_start = mount.get_position()

    while True:
        position = mount.get_position()
        position_change = {}
        for axis in mount.get_axis_names():
            position_change[axis] = track.wrap_error(position[axis] - position_start[axis]) * 3600.0
        print(str(position_change))

if __name__ == "__main__":
    main()
