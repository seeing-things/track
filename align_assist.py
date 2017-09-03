#!/usr/bin/env python

from __future__ import print_function
import config
import configargparse
import track
import mounts
import errorsources
import sys
import time
import math
import numpy as np


parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--camera', help='device name of tracking camera', default='/dev/video0')
parser.add_argument('--camera-res', help='camera resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.1, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=2.0, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.3, type=float)
parser.add_argument('--backlash-az', help='backlash in azimuth (arcseconds)', default=0.0, type=float)
parser.add_argument('--backlash-alt', help='backlash in altitude (arcseconds)', default=0.0, type=float)
parser.add_argument('--align-dir-az', help='azimuth alignment approach direction (-1 or +1)', default=+1, type=int)
parser.add_argument('--align-dir-alt', help='altitude alignment approach direction (-1 or +1)', default=+1, type=int)
args = parser.parse_args()

# Create object with base type TelescopeMount. Disable altitude limits because 
# mount doesn't know its own alitude before alignment is completed.
mount = mounts.NexStarMount(args.scope, alt_min_limit=-180, alt_max_limit=+180)

# Create object with base type ErrorSource
error_source = errorsources.OpticalErrorSource(args.camera, args.camera_res)

tracker = track.Tracker(
    mount = mount, 
    error_source = error_source, 
    update_period = args.loop_period,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

def stop_at_half_frame_callback():
    if tracker.error['az'] is not None:
        if (abs(tracker.error['az']) > HALF_FRAME_ERROR_MAG or 
            abs(tracker.error['alt']) > HALF_FRAME_ERROR_MAG):
                tracker.stop = True

def stop_at_frame_edge_callback():
    if tracker.error['az'] is not None:
        if (abs(tracker.error['az']) > NEAR_FRAME_EDGE_ERROR_MAG or 
            abs(tracker.error['alt']) > NEAR_FRAME_EDGE_ERROR_MAG):
                tracker.stop = True

def error_print_callback():
    if tracker.error['az'] is not None:
        print('\terror (az,alt): ' + str(tracker.error['az'] * 3600.0) + ', ' 
            + str(tracker.error['alt'] * 3600.0))

def stop_beyond_deadband_callback():
    position_stop = mount.get_azalt(remove_backlash=False)
    position_change = abs(errorsources.wrap_error(
        position_stop[other_axis] - position_start[other_axis]
    ))
    if position_change >= backlash:
        tracker.stop = True

def stop_within_max_drift_time():
    if tracker.error[other_axis] is not None:
        time_to_center = abs(tracker.error[other_axis] / apparent_motion[other_axis])
        if time_to_center < MAX_DRIFT_TIME:
            tracker.stop = True

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

def track_until_centered_callback():
    if tracker.error['az'] is not None:
        if (abs(tracker.error['az']) < NEAR_FRAME_CENTER_ERROR_MAG and
            abs(tracker.error['alt']) < NEAR_FRAME_CENTER_ERROR_MAG):
            tracker.stop = True

try:

    # Some of these constants make assumptions about specific hardware
    SLEW_STOP_SLEEP = 1.0
    FAST_SLEW_RATE = 60.0 / 3600.0
    SLOW_SLEW_RATE = 30.0 / 3600.0
    VERY_SLOW_SLEW_RATE = 5.0 / 3600.0
    ANGLE_THRESHOLD = 1.0
    WIDE_ANGLE_THRESHOLD = 10.0
    OPTICAL_ERROR_RETRIES = 10
    MAX_DRIFT_TIME = 10.0
    FRAME_INSCRIBED_DIAMETER_DEG = error_source.degrees_per_pixel * min(
        error_source.frame_height_px, 
        error_source.frame_width_px
    )

    # If the magnitude of the error is larger than this value, the object
    # is more than 50% of the distance from the center of the frame to the
    # nearest edge.
    HALF_FRAME_ERROR_MAG = 0.5 * 0.5 * FRAME_INSCRIBED_DIAMETER_DEG

    # If the magnitude of the error is larger than this value, the object
    # is more than 80% of the distance from the center of the frame to the
    # nearest edge.
    NEAR_FRAME_EDGE_ERROR_MAG = 0.8 * 0.5 * FRAME_INSCRIBED_DIAMETER_DEG

    # If the magnitude of the error is less than this value, the object
    # is less than 10% of the distance from the center of the frame to the
    # nearest edge.
    NEAR_FRAME_CENTER_ERROR_MAG = 0.1 * 0.5 * FRAME_INSCRIBED_DIAMETER_DEG

    print('Centering object...')
    tracker.register_callback(track_until_centered_callback)
    tracker.run()
    tracker.register_callback(None)

    # estimate object's apparent motion with mount stationary
    print('Estimating object apparent motion with mount stationary...', end='')
    mount.slew('az', 0.0)
    mount.slew('alt', 0.0)
    time.sleep(SLEW_STOP_SLEEP)
    error_start = error_source.compute_error(OPTICAL_ERROR_RETRIES)
    time_start = time.time()
    while True:
        error = error_source.compute_error(OPTICAL_ERROR_RETRIES)
        if (abs(error['az']) > HALF_FRAME_ERROR_MAG or 
            abs(error['alt']) > HALF_FRAME_ERROR_MAG):
            break
        elif time.time() - time_start > MAX_DRIFT_TIME:
            break
    error_stop = error_source.compute_error(OPTICAL_ERROR_RETRIES)
    time_elapsed = time.time() - time_start
    apparent_motion = {
        'az': (error_stop['az'] - error_start['az']) / time_elapsed,
        'alt': (error_stop['alt'] - error_start['alt']) / time_elapsed
    }
    apparent_motion_angle = (180.0 / math.pi * math.atan2(
        apparent_motion['alt'], apparent_motion['az']
    )) % 360.0
    track_axes = []
    if ((apparent_motion['az'] > 0.0 and args.align_dir_az == +1) or
        (apparent_motion['az'] < 0.0 and args.align_dir_az == -1)):
        track_axes.append('az')
    if ((apparent_motion['alt'] > 0.0 and args.align_dir_alt == +1) or
        (apparent_motion['alt'] < 0.0 and args.align_dir_alt == -1)):
        track_axes.append('alt')
    print('done.')
    print('\taz apparent motion (arcsec/s): ' + str(apparent_motion['az'] * 3600.0))
    print('\talt apparent motion (arcsec/s): ' + str(apparent_motion['alt'] * 3600.0))
    print('\tapparent motion direction (degrees): ' + str(apparent_motion_angle))
    print('\taxes tracking in approach direction: ' + str(track_axes))

    # case a: Slewing in the desired approach direction in both axes to track
    # object. Just keep doing this indefinitely.
    if len(track_axes) == 2:
        print('Lucky you! Object is moving in approach direction in both axes.')
        print('Waiting for tracking loop to converge...', end='')
        tracker.register_callback(track_until_converged_callback)
        tracker.run(track_axes)
        tracker.register_callback(None)
        print('done.')
        print('Press ALIGN on hand controller at any time. Press CTRL-C to quit.')
        tracker.run(track_axes)

    # case b: One axis is slewing in the desired approach direction but the 
    # other is not.
    elif len(track_axes) == 1:

        print('Object is moving in approach direction in ' + str(*track_axes) + ' axis only.')

        other_axis = 'az' if 'alt' in track_axes else 'alt'
        align_dir = args.align_dir_az if other_axis == 'az' else args.align_dir_alt
        backlash = (args.backlash_az if other_axis == 'az' else args.backlash_alt) / 3600.0

        print('Waiting for tracking loop to converge...', end='')
        tracker.register_callback(track_until_converged_callback)
        tracker.run()
        tracker.register_callback(None)
        print('done.')

        # move object away from center of frame such that its apparent sidereal
        # motion will be towards center
        print('Moving object near edge of frame...', end='')
        mount.slew(other_axis, FAST_SLEW_RATE * -align_dir)
        tracker.register_callback(stop_at_frame_edge_callback)
        tracker.run(track_axes)
        tracker.register_callback(None)
        print('done.')

        # slew in desired approach direction until backlash is removed
        print('Slewing in approach direction past backlash deadband...', end='')
        position_start = mount.get_azalt(remove_backlash=False)
        mount.slew(other_axis, FAST_SLEW_RATE * align_dir)
        tracker.register_callback(stop_beyond_deadband_callback)
        tracker.run(track_axes)
        tracker.register_callback(None)
        print('done.')

        # Continue to slew until object is within about 10 seconds of crossing 
        # frame center. This accelerates the alignment process when the object 
        # is moving very slowly in the 'other' axis.
        print('Waiting until object is within ' + str(MAX_DRIFT_TIME) 
            + ' seconds of frame center...', end='')
        mount.slew(other_axis, SLOW_SLEW_RATE * align_dir)
        tracker.register_callback(stop_within_max_drift_time)
        tracker.run(track_axes)
        tracker.register_callback(None)
        print('done.')

        # wait for object to drift back towards center.
        print('Press ALIGN on hand controller when object crosses frame center.')
        mount.slew(other_axis, 0.0)
        tracker.register_callback(error_print_callback)
        tracker.run(track_axes)
        tracker.register_callback(None)

    # case c: Neither axis is slewing in the desired approach direction
    else:

        print('Centering object...', end='')
        tracker.register_callback(track_until_centered_callback)
        tracker.run()
        tracker.register_callback(None)
        print('done.')

        # move object away from center of frame such that its apparent sidereal
        # motion will be towards center
        print('Moving object near edge of frame...', end='')
        mount.slew('az', FAST_SLEW_RATE * -args.align_dir_az)
        mount.slew('alt', FAST_SLEW_RATE * -args.align_dir_alt)
        while True:
            error = error_source.compute_error(OPTICAL_ERROR_RETRIES)
            if (abs(error['az']) > NEAR_FRAME_EDGE_ERROR_MAG or 
                abs(error['alt']) > NEAR_FRAME_EDGE_ERROR_MAG):
                break
        print('done.')

        # slew in desired approach direction until backlash is removed
        print('Slewing in approach direction past backlash deadband...', end='')
        position_start = mount.get_azalt(remove_backlash=False)
        mount.slew('az', FAST_SLEW_RATE * args.align_dir_az)
        mount.slew('alt', FAST_SLEW_RATE * args.align_dir_alt)
        while True:
            # calling this so camera output is displayed, output not used
            error_source.compute_error()
            position = mount.get_azalt(remove_backlash=False)
            position_change = {
                'az': abs(errorsources.wrap_error(position['az'] - position_start['az'])),
                'alt': abs(errorsources.wrap_error(position['alt'] - position_start['alt'])),
            }
            if position_change['az'] >= args.backlash_az / 3600.0:
                mount.slew('az', 0.0)
            if position_change['alt'] >= args.backlash_alt / 3600.0:
                mount.slew('alt', 0.0)
            if ((position_change['az'] >= args.backlash_az / 3600.0) and
                (position_change['alt'] >= args.backlash_alt / 3600.0)):
                break
        print('done.')
        
        # move object such that its apparent motion vector intersects the
        # center of the frame
        print('Moving object such that velocity vector intersects center of frame...')
        error = error_source.compute_error(OPTICAL_ERROR_RETRIES)
        error_angle = 180.0 / math.pi * math.atan2(error['alt'], error['az'])
        obj_to_center_angle = (error_angle + 180.0) % 360.0
        print('\tangle from object to center frame: ' + str(obj_to_center_angle))
        if ((obj_to_center_angle >= 0.0 and obj_to_center_angle < 90.0) or 
            (obj_to_center_angle >= 180.0 and obj_to_center_angle < 270.0)):
            if obj_to_center_angle > apparent_motion_angle:
                active_axis = 'alt'
            else:
                active_axis = 'az'
        else:
            if apparent_motion_angle > obj_to_center_angle:
                active_axis = 'alt'
            else:
                active_axis = 'az'
        print('\tslewing in ' + active_axis)
        active_axis_align_dir = args.align_dir_alt if active_axis == 'alt' else args.align_dir_az
        mount.slew(active_axis, SLOW_SLEW_RATE * active_axis_align_dir)
        angle_diff_prev = errorsources.wrap_error(obj_to_center_angle - apparent_motion_angle)
        very_slow = False
        while True:
            error = error_source.compute_error(OPTICAL_ERROR_RETRIES)
            error_angle = 180.0 / math.pi * math.atan2(error['alt'], error['az'])
            obj_to_center_angle = (error_angle + 180.0) % 360.0
            angle_diff = errorsources.wrap_error(obj_to_center_angle - apparent_motion_angle)
            print('\tangle diff: ' + str(angle_diff))
            if ((np.sign(angle_diff) != np.sign(angle_diff_prev)) or 
                (abs(angle_diff) < ANGLE_THRESHOLD)):
                mount.slew('az', 0.0)
                mount.slew('alt', 0.0)
                break
            elif abs(angle_diff) < WIDE_ANGLE_THRESHOLD and very_slow == False:
                print('\tnearing correct angle, very slow slew rate enabled')
                very_slow = True
                mount.slew(active_axis, VERY_SLOW_SLEW_RATE * active_axis_align_dir)
            angle_diff_prev = angle_diff
        print('Done.')

        # wait for object to drift back towards center.
        print('Press ALIGN on hand controller when object crosses frame center...')
        while True:
            error = error_source.compute_error(OPTICAL_ERROR_RETRIES)
            print('\terror (az,alt): ' + str(error['az'] * 3600.0) + ', ' 
                + str(error['alt'] * 3600.0))

    mount.slew('az', 0.0)
    mount.slew('alt', 0.0)

except KeyboardInterrupt:
    print('Goodbye!')
    pass
