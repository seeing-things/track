#!/usr/bin/env python

import track
import mounts
import error
import math
import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument('--camera', help='device name of tracking camera', default='/dev/video0')
parser.add_argument('--camera-res', help='camera resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

# Create object with base type ErrorSource
error_source = error.OpticalErrorSource(args.camera, args.camera_res)

tracker = track.Tracker(
    mount = mount, 
    error_source = error_source, 
    update_period = 0.25,
    loop_bandwidth = 0.5,
    damping_factor = math.sqrt(2.0) / 2.0
)
tracker.start()

try:
    print('Press Enter to quit.')
    raw_input()
except KeyboardInterrupt:
    pass
finally:
    tracker.stop()
    print('Goodbye!')
