#!/usr/bin/env python

import track
import mounts
import error
import argparse
import time
import numpy as np

class TrackUntilConverged(track.Tracker):

    ERROR_THRESHOLD = 50.0 / 3600.0
    MIN_ITERATIONS = 5

    def _stopping_condition(self):
        try:
            if abs(self.error_az) > self.ERROR_THRESHOLD or abs(self.error_alt) > self.ERROR_THRESHOLD:
                self.low_error_iterations = 0
                return False
        except TypeError:
            return False
        
        try:
            self.low_error_iterations += 1
        except AttributeError:
            self.low_error_iterations = 1

        if self.low_error_iterations >= self.MIN_ITERATIONS:
            return True
        else:
            return False


parser = argparse.ArgumentParser()
parser.add_argument('--camera', help='device name of tracking camera', default='/dev/video0')
parser.add_argument('--camera-res', help='camera resolution in arcseconds per pixel', required=True, type=float)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.1, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=2.0, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.5, type=float)
args = parser.parse_args()

# Create object with base type TelescopeMount
mount = mounts.NexStarMount(args.scope)

# Create object with base type ErrorSource
error_source = error.OpticalErrorSource(args.camera, args.camera_res)

tracker = TrackUntilConverged(
    mount = mount, 
    error_source = error_source, 
    update_period = args.loop_period,
    loop_bandwidth = args.loop_bw,
    damping_factor = args.loop_damping
)

try:

    MOVEMENT_THRESHOLD_DEG = 20.0 / 3600.0
    SLEW_STOP_SLEEP = 1.0
    FAST_SLEW_RATE = 60.0 / 3600.0
    SLOW_SLEW_RATE = 30.0 / 3600.0
    MIN_INCREMENT = 1.0 / 3600.0
    NUM_ITERATIONS = 5

    backlash_estimates = {}
    for axis in ['az', 'alt']:

        backlash_estimates[axis] = []

        for i in range(NUM_ITERATIONS):

            print('Start iteration ' + str(i) + ' of ' + str(NUM_ITERATIONS))
            print('Centering object in FOV...')

            # center the object in the FOV
            tracker.low_error_iterations = 0
            tracker.run()
            
            # Mount is still slewing until commanded to do otherwise.
            # Intentionally allowing the axis not under test to continue to slew
            # at the tracking rate so the object is less likely to drift out 
            # of the FOV on that axis during the test.

            print('Object has been centered. Slewing in ' + axis + ' until movement detected...')

            # slew in axis under test until movement is detected
            mount.slew(axis, FAST_SLEW_RATE)
            while True:
                error = error_source.compute_error()
                if abs(error[axis]) > MOVEMENT_THRESHOLD_DEG:
                    mount.slew(axis, 0.0)
                    time.sleep(SLEW_STOP_SLEEP)
                    break

            # record position
            init_error = error_source.compute_error()
            init_az, init_alt = mount.get_azalt()

            print('Movement detected. Optical position error is ' + str(init_error[axis] * 3600.0) + ' arcseconds')
            print('Slewing in other direction until movement detected...')

            # slew the other direction until movement is detected
            mount.slew(axis, -SLOW_SLEW_RATE)
            while True:
                error = error_source.compute_error()
                if abs(error[axis] - init_error[axis]) > MOVEMENT_THRESHOLD_DEG:
                    az, alt = mount.get_azalt()
                    mount.slew(axis, 0.0)
                    break

            if axis == 'az':
                backlash_estimates[axis].append(abs(error.wrap_error(az - init_az)))
            elif axis == 'alt':
                backlash_estimates[axis].append(abs(error.wrap_error(alt - init_alt)))
            else:
                raise ValueError('axis value is not az or alt')

            print('Movement detected. Estimate of backlash is ' + str(backlash_estimates[axis][-1] * 3600.0) + ' arcseconds.')

        print('Iterations for ' + axis + ' axis completed.')
        print('Mean backlash: ' + str(np.mean(backlash_estimates)) + ' arcseconds')
        print('Standard deviation: ' + str(np.std(backlash_estimates)) + ' arcseconds')

    # stop the mount
    for axis in axes:
        mount.slew(axis, 0.0)

except KeyboardInterrupt:
    print('Goodbye!')
    pass
