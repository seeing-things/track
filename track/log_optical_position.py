#!/usr/bin/env python

"""Appends mount-reported position to a file.

This intent of this script is to append the positions of a variety of celestial objects to a file
for post processing. The assumption is that the position data corresponds to raw encoder readings
without any corrections or alignment. This data can then be processed to develop a model for the
mount and to assess systematic errors that may reflect deficiencies in the mount hardware.
"""

from __future__ import print_function
import time
import datetime
import math
import ephem
import track


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

parser = track.ArgParser()
parser.add_argument(
    '--lat',
    required=True,
    help='latitude of observer (+N)')
parser.add_argument(
    '--lon',
    required=True,
    help='longitude of observer (+E)')
parser.add_argument(
    '--elevation',
    required=True,
    help='elevation of observer (m)',
    type=float)
parser.add_argument(
    '--camera',
    help='device node path for tracking webcam',
    default='/dev/video0')
parser.add_argument(
    '--camera-res',
    help='webcam resolution in arcseconds per pixel',
    required=True,
    type=float)
parser.add_argument(
    '--camera-bufs',
    help='number of webcam capture buffers',
    required=True,
    type=int)
parser.add_argument(
    '--camera-exposure',
    help='webcam exposure level',
    default=2000,
    type=int)
parser.add_argument(
    '--scope',
    help='serial device for connection to telescope',
    default='/dev/ttyUSB0')
parser.add_argument(
    '--loop-bw',
    help='control loop bandwidth (Hz)',
    default=0.5,
    type=float)
parser.add_argument(
    '--loop-damping',
    help='control loop damping factor',
    default=2.0,
    type=float)
parser.add_argument(
    '--filename',
    help='output log filename',
    default='object_data')
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = track.NexStarMount(args.scope)

# Create object with base type ErrorSource
error_source = track.OpticalErrorSource(
    args.camera,
    args.camera_res,
    args.camera_bufs,
    args.camera_exposure,
    x_axis_name='az',
    y_axis_name='alt'
)

tracker = track.Tracker(
    mount=mount,
    error_source=error_source,
    loop_bandwidth=args.loop_bw,
    damping_factor=args.loop_damping
)

print('Tracking object until converged...')
tracker.register_callback(track_until_converged_callback)
tracker.run()

now = time.time()
position = mount.get_position()
directions = {}
for axis in mount.get_axis_names():
    directions[axis] = +1 if tracker.loop_filter[axis].int > 0 else -1

print('Time: ' + str(now))
print('Position: ' + str(position))
print('Directions: ' + str(directions))
object_name = raw_input('Name of object: ')

# try to find object in database to get true coordinates
try:
    target = ephem.star(object_name)
except KeyError:
    ss_objs = [n for _0, _1, n in ephem._libastro.builtin_planets()]
    if object_name in ss_objs:
        body_type = getattr(ephem, object_name)
        target = body_type()
    else:
        target = None
        print('Could not find ' + object_name + ' in database')

if target is not None:
    observer = ephem.Observer()
    observer.lat = args.lat
    observer.lon = args.lon
    observer.elevation = args.elevation
    observer.date = ephem.Date(datetime.datetime.utcfromtimestamp(now))
    target.compute(observer)
    actual_position = {
        'az': float(target.az) * 180 / math.pi,
        'alt': float(target.alt) * 180 / math.pi
    }
else:
    actual_position = {
        'az': float('nan'),
        'alt': float('nan')
    }

with open(args.filename, 'a') as f:
    f.write(','.join([
        object_name,
        str(now),
        str(position['az']),
        str(position['alt']),
        str(directions['az']),
        str(directions['alt']),
        str(actual_position['az']),
        str(actual_position['alt']),
    ]) + '\n')
