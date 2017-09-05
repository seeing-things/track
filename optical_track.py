#!/usr/bin/env python

import config
import configargparse
import track
import mounts
import errorsources

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--camera', help='device node path for tracking webcam', default='/dev/video0')
parser.add_argument('--camera-res', help='webcam resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--camera-w', help='desired webcam capture width in pixels', required=True, type=int)
parser.add_argument('--camera-h', help='desired webcam capture height in pixels', required=True, type=int)
parser.add_argument('--camera-bufs', help='number of webcam capture buffers', required=True, type=int)
parser.add_argument('--camera-exposure', help='webcam exposure level', default=2000, type=int)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.1, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=2.0, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.3, type=float)
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

# Create object with base type ErrorSource
error_source = errorsources.OpticalErrorSource(args.camera, args.camera_res, (args.camera_w, args.camera_h), args.camera_bufs, args.camera_exposure)

tracker = track.Tracker(
    mount = mount, 
    error_source = error_source, 
    update_period = args.loop_period,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

try:
    tracker.run()
except KeyboardInterrupt:
    print('Goodbye!')
    pass
