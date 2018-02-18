#!/usr/bin/env python

# This intent of this script is to append the positions of a variety of
# celestial objects to a file for post processing. The assumption is that the
# position data corresponds to raw encoder readings without any corrections
# or alignment. This data can then be processed to develop a model for the
# mount and to assess systematic errors that may reflect deficiencies in the
# mount hardware.

import config
import configargparse
import mounts
import errorsources
import track
import time


def track_until_converged_callback():
    ERROR_THRESHOLD = 20.0 / 3600.0
    MIN_ITERATIONS = 50

    try:
        if (abs(tracker.error['az']) > ERROR_THRESHOLD or
            abs(tracker.error['alt']) > ERROR_THRESHOLD):
            tracker.low_error_iterations = 0
            return
    except TypeError:
        return

    if hasattr(tracker, 'low_error_iterations'):
        tracker.low_error_iterations += 1
    else:
        tracker.low_error_iterations = 1

    if tracker.low_error_iterations >= MIN_ITERATIONS:
        tracker.low_error_iterations = 0
        tracker.stop = True

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--camera', help='device node path for tracking webcam', default='/dev/video0')
parser.add_argument('--camera-res', help='webcam resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--camera-bufs', help='number of webcam capture buffers', required=True, type=int)
parser.add_argument('--camera-exposure', help='webcam exposure level', default=2000, type=int)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.5, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=2.0, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.3, type=float)
parser.add_argument('--filename', help='output log filename', default='object_data')
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

# Create object with base type ErrorSource
error_source = errorsources.OpticalErrorSource(
    args.camera,
    args.camera_res,
    args.camera_bufs,
    args.camera_exposure
)

tracker = track.Tracker(
    mount = mount,
    error_source = error_source,
    update_period = args.loop_period,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

print('Tracking object until converged...')
tracker.register_callback(track_until_converged_callback)
tracker.run()

now = time.time()
position = mount.get_azalt()
directions = {}
for d in ['az', 'alt']:
    directions[d] = +1 if tracker.loop_filter[d].int > 0 else -1

print('Time: ' + str(now))
print('Position: ' + str(position))
print('Directions: ' + str(directions))
raw_input('Name of object: ')

with open(args.filename, 'a') as f:
    f.write(','.join([
        object_name,
        str(now),
        str(position['az']),
        str(position['alt']),
        str(directions['az']),
        str(directions['alt'])
    ]) + '\n')
