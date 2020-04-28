#!/usr/bin/env python3

"""Computer vision tracking of objects.

This program uses a feedback control loop to cause a telescope mount to track an object based on
computer vision detection of that object in a camera. The distance of the object from the center
of the camera frame is used as an error term such that the control loop will attempt to center
the object in the frame.

An optional gamepad can be used to manually control the mount when no targets are in view. When
a target is acquired by the camera gamepad control is inhibited.
"""

import os
import sys
import click
import track
from track import cameras

def main():
    """See module docstring"""

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
    cameras.add_program_arguments(parser, profile='track')
    args = parser.parse_args()

    # Set priority of this thread to realtime. Do this before constructing objects since priority
    # is inherited and some critical threads are created by libraries we have no direct control
    # over.
    os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(11))

    def gamepad_callback(tracker: track.Tracker) -> bool:
        """Callback for gamepad control.

        Allows manual control of the slew rate via a gamepad when the 'B' button is held down,
        overriding optical tracking behavior. This callback is registered with the Tracker object
        which calls it near the start of each control cycle.

        Args:
            tracker: A reference to an object of type Tracker. Not used internally.

        Returns:
            True when the 'B' button is depressed. False otherwise.
        """
        if game_pad.state.get('BTN_EAST', 0) == 1:
            gamepad_x, gamepad_y = game_pad.get_proportional()
            slew_rates = (
                mount.max_slew_rate * gamepad_x,
                mount.max_slew_rate * gamepad_y
            )
            for idx, axis_name in enumerate(mount.AxisName):
                tracker.slew_rate[axis_name], _ = mount.slew(axis_name, slew_rates[idx])
                # Set loop filter integrators to match so that when gamepad control is released
                # there is a smooth handoff to optical tracking control.
                tracker.controllers[axis_name].integrator = tracker.slew_rate[axis_name]
            return True
        else:
            return False

    # Create object with base type TelescopeMount
    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path, bypass_alt_limits=args.bypass_alt_limits)
        if args.bypass_alt_limits:
            print('Warning: Altitude limits disabled! Be careful!')
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)

    # Load a MountModel object
    try:
        mount_model = track.model.load_stored_model()
    except track.model.StaleParametersException:
        if click.confirm('Stored model parameters are stale. Use anyway?', default=False):
            mount_model = track.model.load_stored_model(max_age=None)
        elif click.confirm('Okay. Use a default set of model parameters instead?', default=False):
            mount_model = track.model.load_default_model()
        else:
            print('No valid mount model could be loaded. Aborting.')
            sys.exit(1)

    # Create object with base type ErrorSource
    camera = cameras.make_camera_from_args(args, profile='track')
    error_source = track.OpticalErrorSource(
        camera=camera,
        mount=mount,
        mount_model=mount_model,
    )
    telem_sources = {'error_optical': error_source}

    tracker = track.Tracker(
        mount=mount,
        error_source=error_source,
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
        print('Got CTRL-C, shutting down...')
    except Exception as e:
        print('Unhandled exception: ' + str(e))
        import traceback
        traceback.print_exc()
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...', end='', flush=True)
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

        if args.telem_enable:
            telem_logger.stop()

        try:
            game_pad.stop()
        except:
            pass

if __name__ == "__main__":
    main()
