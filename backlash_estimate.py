#!/usr/bin/env python

import track
import mounts
import error
import argparse

class TrackUntilConverged(track.Tracker):

    ERROR_THRESHOLD = 0.01
    MIN_ITERATIONS = 40

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
parser.add_argument('--loop-bw', help='control loop bandwidth (Hz)', default=0.5, type=float)
parser.add_argument('--loop-damping', help='control loop damping factor', default=0.5, type=float)
parser.add_argument('--loop-period', help='control loop period', default=0.25, type=float)
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
    SLEW_STOP_SLEEP = 1
    SLEW_RATE_ARCSECS_PER_SEC = 60.0
    SLEW_RATE_DEG_PER_SEC = SLEW_RATE_ARCSECS_PER_SEC / 3600.0
    MIN_INCREMENT = 1.0 / 3600.0

    """
    Establish upper bound on the backlash
    """

    print('Attempting to center object in FOV')

    # center the object in the FOV
    tracker.run()
    mount.slew(0, 0)

    print('Object has been centered. Slewing in azimuth until movement detected...')

    # slew in azimuth until movement is detected
    mount.slew(SLEW_RATE_DEG_PER_SEC, 0)
    while True:
        error_az, error_alt = error_source.compute_error()
        if abs(error_az) > MOVEMENT_THRESHOLD_DEG:
            mount.slew(0, 0)
            time.sleep(SLEW_STOP_SLEEP)
            break

    # record position
    init_error_az, init_error_alt = error_source.compute_error()
    init_az, init_alt = mount.get_azel()

    print('Movement detected. Optical azimuth position error is ' + str(init_error_az * 3600.0) + ' arcseconds')
    print('Slewing in other direction until movement detected...')

    # slew in azimuth the other direction until movement is detected
    mount.slew(-SLEW_RATE_DEG_PER_SEC, 0)
    while True:
        error_az, error_alt = error_source.compute_error()
        if abs(error_az - init_error_az) > MOVEMENT_THRESHOLD_DEG:
            mount.slew(0, 0)
            time.sleep(SLEW_STOP_SLEEP)
            break

    az, alt = mount.get_azel()
    backlash_upper_bound = abs(az - init_az)

    print('Movement detected. Upper bound on azimuth backlash is ' + str(backlash_upper_bound * 3600.0) + ' arcseconds.')
    print('Starting binary search for true backlash value.')

    """
    Binary search for the true backlash
    """
    backlash_test_val = backlash_upper_bound / 2.0
    backlash_increment = backlash_test_val / 2.0

    while True:

        print('Start of iteration. Backlash test value: ' + str(backlash_test_val * 3600.0) + ' arcseconds.')
        print('Centering object in FOV...')

        # center the object in the FOV
        tracker.run()
        mount.slew(0, 0)

        print('Object centered. Slewing until movement detected...')

        # slew in azimuth until movement is detected
        mount.slew(SLEW_RATE_DEG_PER_SEC, 0)
        while True:
            error_az, error_alt = error_source.compute_error()
            if abs(error_az) > MOVEMENT_THRESHOLD_DEG:
                mount.slew(0, 0)
                time.sleep(SLEW_STOP_SLEEP)
                break

        # record starting position
        init_error_az, init_error_alt = error_source.compute_error()
        init_az, init_alt = mount.get_azel()

        print('Movement detected. Slewing in opposite direction by test amount...')

        # slew in other direction by test amount
        mount.slew(-SLEW_RATE_DEG_PER_SEC, 0)
        while True:
            az, alt = mount.get_azel()
            if abs(az - init_az) >= backlash_test_val:
                mount.slew(0, 0)
                break

        print('Mount reports that it moved ' + str(abs(az - init_az) * 3600.0) + ' arcseconds.')
        print('Optical change in target position was ' + str(abs(error_az - init_error_az) * 3600.0) + ' arcseconds')

        error_az, error_alt = error_source.compute_error()

        if abs(error_az - init_error_az) > MOVEMENT_THRESHOLD_DEG:
            print('Movement detected. Decreasing backlash test value by ' + str(backlash_increment * 3600.0) + ' arcseconds.')
            backlash_test_val -= backlash_increment
        else:
            print('Movement not detected. Increasing backlash test value by ' + str(backlash_increment * 3600.0) + ' arcseconds.')
            backlash_test_val += backlash_increment

        backlash_increment /= 2.0

        if backlash_increment < MIN_INCREMENT:
            backlash_az = backlash_test_val
            break

    print('Search completed. Backlash estimated as ' + str(backlash_az * 3600.0) + ' arcseconds.')

except KeyboardInterrupt:
    print('Goodbye!')
    pass
