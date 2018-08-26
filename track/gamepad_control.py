#!/usr/bin/env python3

"""Direct gamepad control of telescope mount.

This program allows direct control of a telescope mount with a gamepad. The slew rates of the mount
are controlled by the analog sticks.
"""

from __future__ import print_function
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
    parser.add_argument(
        '--bypass-alt-limits',
        help='bypass mount altitude limits',
        action='store_true'
    )
    parser.add_argument(
        '--telem-enable',
        help='enable logging of telemetry to database',
        action='store_true'
    )
    parser.add_argument(
        '--telem-db-host',
        help='hostname of InfluxDB database server',
        default='localhost'
    )
    parser.add_argument(
        '--telem-db-port',
        help='port number of InfluxDB database server',
        default=8086,
        type=int
    )
    parser.add_argument(
        '--telem-period',
        help='telemetry sampling period in seconds',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--laser-ftdi-serial',
        help='serial number of laser pointer FTDI device',
    )
    args = parser.parse_args()

    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path, bypass_alt_limits=args.bypass_alt_limits)
        if args.bypass_alt_limits:
            print('Warning: Altitude limits disabled! Be careful!')
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        mount = None
        print('mount-type not supported: ' + args.mount_type)

    game_pad = track.Gamepad()

    try:
        laser = track.LaserPointer(serial_num=args.laser_ftdi_serial)
        game_pad.register_callback('BTN_SOUTH', laser.set)
    except OSError:
        print('Could not connect to laser pointer control device.')
        laser = None

    if args.telem_enable:
        telem_logger = track.TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources={
                'gamepad': game_pad
            }
        )
        telem_logger.start()

    try:
        while True:
            if mount is not None:
                x, y = game_pad.get_value()
                mount.slew('ra', mount.max_slew_rate * x)
                mount.slew('dec', mount.max_slew_rate * y)
    except KeyboardInterrupt:
        print('Goodbye!')
    except Exception as e:
        print('Unhandled exception: ' + str(e))

    if args.telem_enable:
        telem_logger.stop()
        game_pad.stop()

if __name__ == "__main__":
    main()
