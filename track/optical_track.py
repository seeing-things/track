#!/usr/bin/env python3

"""Computer vision tracking of objects.

This program uses a feedback control loop to cause a telescope mount to track an object based on
computer vision detection of that object in a camera. The distance of the object from the center
of the camera frame is used as an error term such that the control loop will attempt to center
the object in the frame.

An optional gamepad can be used to manually control the mount when no targets are in view. When
a target is acquired by the camera gamepad control is inhibited.
"""

from __future__ import print_function
import sys
import track

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--camera',
        help='device node path for tracking webcam',
        default='/dev/video0'
    )
    parser.add_argument(
        '--camera-res',
        help='webcam resolution in arcseconds per pixel',
        required=True,
        type=float
    )
    parser.add_argument(
        '--camera-bufs',
        help='number of webcam capture buffers',
        required=True,
        type=int
    )
    parser.add_argument(
        '--camera-exposure',
        help='webcam exposure level',
        default=3200,
        type=int
    )
    parser.add_argument(
        '--frame-dump-dir',
        help='directory to save webcam frames as jpeg files on disk',
    )
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
        '--loop-bw',
        help='control loop bandwidth (Hz)',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--loop-damping',
        help='control loop damping factor',
        default=2.0,
        type=float
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

    def gamepad_callback(tracker):
        """Callback for gamepad control.

        Allows manual control of the slew rate via a gamepad when the control loop is not tracking
        any optical targets. Gamepad control is inhibited when a target is in view. This callback
        is registered with the Tracker object which calls it on every control cycle.
        """

        # don't try to fight the control loop
        if not any(tracker.error.values()):
            gamepad_x, gamepad_y = game_pad.get_proportional()
            mount.slew(x_axis_name, mount.max_slew_rate * gamepad_x)
            mount.slew(y_axis_name, mount.max_slew_rate * gamepad_y)

    # Create object with base type TelescopeMount
    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path, bypass_alt_limits=args.bypass_alt_limits)
        if args.bypass_alt_limits:
            print('Warning: Altitude limits disabled! Be careful!')
        x_axis_name = 'az'
        y_axis_name = 'alt'
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
        x_axis_name = 'ra'
        y_axis_name = 'dec'
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    # Create object with base type ErrorSource
    error_source = track.OpticalErrorSource(
        cam_dev_path=args.camera,
        arcsecs_per_pixel=args.camera_res,
        cam_num_buffers=args.camera_bufs,
        cam_ctlval_exposure=args.camera_exposure,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        mount=mount,
        frame_dump_dir=args.frame_dump_dir,
    )
    telem_sources = {'error_optical': error_source}

    tracker = track.Tracker(
        mount=mount,
        error_source=error_source,
        loop_bandwidth=args.loop_bw,
        damping_factor=args.loop_damping
    )
    telem_sources['tracker'] = tracker

    try:
        laser = track.LaserPointer(serial_num=args.laser_ftdi_serial)
    except OSError:
        print('Could not connect to laser pointer FTDI device.')
        laser = None

    try:
        game_pad = track.Gamepad()
        if laser is not None:
            game_pad.register_callback('BTN_SOUTH', laser.set)
        tracker.register_callback(gamepad_callback)
        telem_sources['gamepad'] = game_pad
        print('Gamepad found and registered.')
    except RuntimeError:
        print('No gamepads found.')

    if args.telem_enable:
        telem_logger = track.TelemLogger(
            host=args.telem_db_host,
            port=args.telem_db_port,
            period=args.telem_period,
            sources=telem_sources,
        )
        telem_logger.start()

    try:
        tracker.run()
    except KeyboardInterrupt:
        print('Goodbye!')
    except Exception as e:
        print('Unhandled exception: ' + str(e))
    finally:
        # don't rely on destructors to safe mount!
        mount.safe()

    if args.telem_enable:
        telem_logger.stop()

    try:
        game_pad.stop()
    except:
        pass

if __name__ == "__main__":
    main()
