#!/usr/bin/env python3

"""Automated tracking of objects.

This is the core program in this project. The software uses a control system to cause a telescope
mount to track a target based on the target's position in the sky and the mount's encoder readings.
"""
import os
import sys
import click
import astropy.units as u
from astropy.coordinates import Latitude, Longitude
import track
from track import laser, mounts, targets, telem
from track.control import Tracker
from track.gamepad import Gamepad
from track.mounts import MeridianSide


def main():
    """See module docstring"""

    def target_nudge(stick, state):
        if stick == 'updown':
            if state == -1:
                target.nudge('up')
            elif state == 1:
                target.nudge('down')
        elif stick == 'leftright':
            if state == -1:
                target.nudge('left')
            elif state == 1:
                target.nudge('right')

    def target_nudge_clear(state):
        if state == 1:
            target.nudge_clear()

    def gamepad_callback(tracker: Tracker) -> bool:
        """Callback for gamepad control.

        Allows manual control of the slew rate via a gamepad when the 'B' button is held down,
        overriding tracking behavior. This callback is registered with the Tracker object which
        calls it on each control cycle.

        Defined inside main() so this has easy access to objects that are within that scope.

        Args:
            tracker: A reference to an object of type Tracker. Not used internally.

        Returns:
            True when the 'B' button is depressed which disables the normal control path. False
                otherwise, which leaves the normal control path enabled.
        """
        if game_pad.state.get('BTN_EAST', 0) == 1:
            gamepad_x, gamepad_y = game_pad.get_proportional()
            slew_rates = (
                mount.max_slew_rate * gamepad_x,
                mount.max_slew_rate * gamepad_y
            )
            for idx, axis_name in enumerate(mount.AxisName):
                mount.slew(axis_name, slew_rates[idx])
            return True
        else:
            return False


    parser = track.ArgParser()
    targets.add_program_arguments(parser)
    laser.add_program_arguments(parser)
    mounts.add_program_arguments(parser)
    telem.add_program_arguments(parser)
    args = parser.parse_args()

    # Set priority of this thread to realtime. Do this before constructing objects since priority
    # is inherited and some critical threads are created by libraries we have no direct control
    # over.
    os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(11))


    # Load a MountModel object
    try:
        mount_model = track.model.load_stored_model()
    except track.model.StaleParametersException:
        if click.confirm('Stored alignment parameters are stale. Use anyway?', default=False):
            mount_model = track.model.load_stored_model(max_age=None)
        elif args.target_type == 'camera':
            if click.confirm('Use a default set of alignment parameters instead?', default=True):
                guide_cam_orientation = click.prompt(
                    'Enter guide camera orientation in degrees, clockwise positive', type=float)
                mount_model = track.model.load_default_model(
                    guide_cam_orientation=Longitude(guide_cam_orientation*u.deg))

    if 'mount_model' not in locals():
        print('Aborting: No model could be loaded. To refresh stored model run align program.')
        sys.exit(1)

    # Create object with base type TelescopeMount
    mount = mounts.make_mount_from_args(args)

    target = targets.make_target_from_args(
        args,
        mount,
        mount_model,
        MeridianSide[args.meridian_side.upper()]
    )

    telem_sources = {}
    tracker = Tracker(
        mount=mount,
        mount_model=mount_model,
        target=target,
    )
    telem_sources['tracker'] = tracker

    try:
        laser_pointer = laser.make_laser_from_args(args)
    except OSError:
        print('Could not connect to laser pointer FTDI device.')
        laser_pointer = None

    try:
        # Create gamepad object and register callback
        game_pad = Gamepad()
        if laser_pointer is not None:
            game_pad.register_callback('BTN_SOUTH', laser_pointer.set)

        # D-pad adjusts target position
        game_pad.register_callback('ABS_HAT0Y', lambda state: target_nudge('updown', state))
        game_pad.register_callback('ABS_HAT0X', lambda state: target_nudge('leftright', state))
        game_pad.register_callback('BTN_SOUTH', target_nudge_clear)

        tracker.register_callback(gamepad_callback)
        telem_sources['gamepad'] = game_pad
        print('Gamepad found and registered.')
    except RuntimeError:
        print('No gamepads found.')

    if args.telem_enable:
        telem_logger = telem.make_telem_logger_from_args(args, telem_sources)
        telem_logger.start()

    try:
        tracker.run()
    except KeyboardInterrupt:
        print('Got CTRL-C, shutting down...')
    finally:
        # don't rely on destructors to safe mount!
        print('Safing mount...', end='', flush=True)
        if mount.safe():
            print('Mount safed successfully!')
        else:
            print('Warning: Mount may be in an unsafe state!')

        try:
            telem_logger.stop()
        except UnboundLocalError:
            pass

        try:
            game_pad.stop()
        except UnboundLocalError:
            pass

if __name__ == "__main__":
    main()
