#!/usr/bin/env python

import config
import configargparse
import track
import mounts
import errorsources
import gamepad

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--camera', help='device node path for tracking webcam', default='/dev/video0')
parser.add_argument('--camera-res', help='webcam resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--camera-bufs', help='number of webcam capture buffers', required=True, type=int)
parser.add_argument('--camera-exposure', help='webcam exposure level', default=2000, type=int)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.5, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=2.0, type=float)
parser.add_argument('--bypass-alt-limits', help='bypass mount altitude limits', action='store_true')
args = parser.parse_args()

# this callback allows manual control of the slew rate via a gamepad when the
# control loop is not tracking any optical targets
def gamepad_callback():
    # don't try to fight the control loop
    if tracker.error['az'] == None:
        try:
            x, y = game_pad.get_proportional()
            mount.slew('az', mount.max_slew_rate * x)
            mount.slew('alt', mount.max_slew_rate * y)
        except mount.AltitudeLimitException:
            pass

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope, bypass_alt_limits=args.bypass_alt_limits)
if args.bypass_alt_limits:
    print('Warning: Altitude limits disabled! Be careful!')

# Create object with base type ErrorSource
error_source = errorsources.OpticalErrorSource(args.camera, args.camera_res, args.camera_bufs, args.camera_exposure)

tracker = track.Tracker(
    mount = mount,
    error_source = error_source,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

try:
    game_pad = gamepad.Gamepad()
    tracker.register_callback(gamepad_callback)
    print('Gamepad found and registered.')
except RuntimeError:
    print('No gamepads found.')

try:
    tracker.run()
except KeyboardInterrupt:
    print('Goodbye!')
    pass
