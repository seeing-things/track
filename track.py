#!/usr/bin/env python

import sys
import ephem
import time
import datetime
import math
import nexstar
import threading
import argparse
import abc

# return value limited to the range [-limit,+limit]
def clamp(x, limit):
    return max(min(limit, x), -limit)

# wraps a phase angle in degrees to the range [-180,+180)
def wrap_error(e):
    return (e + 180.0) % 360.0 - 180.0

class ErrorSource:
    __metaclass__ = abc.ABCMeta

    # returns the pointing error as a tuple of (az, alt) in degrees where the 
    # azimuth range is [0,360) and the altitude range is [-180,180)
    @abc.abstractmethod
    def compute_error(self):
        pass

class BlindErrorSource(ErrorSource):

    def __init__(self, mount, observer, target):

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # NexStar object
        self.mount = mount

    def compute_error(self):

        # get current coordinates of the target in degrees
        # not using ephem.now() because it rounds time to the nearest second
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_az_deg = self.target.az * 180.0 / math.pi
        target_alt_deg = self.target.alt * 180.0 / math.pi

        # get current position of telescope (degrees)
        (scope_az_deg, scope_alt_deg) = self.mount.get_azel()
         
        # compute pointing errors in degrees
        error_az = wrap_error(target_az_deg - scope_az_deg)
        error_alt = wrap_error(target_alt_deg - scope_alt_deg)

        return (error_az, error_alt)

# Two independent proportional plus integral (PI) loop filters, one for 
# azimuth and another for altitude.
class LoopFilter:
    def __init__(self, bandwidth, damping_factor, update_period, rate_limit):
        
        # compute loop filter gains
        bt = bandwidth * update_period
        k0 = update_period
        denom = damping_factor + 1.0 / (4.0 * damping_factor)
        self.prop_gain = 4.0 * damping_factor / denom * bt / k0
        self.int_gain = 4.0 / denom**2.0 * bt**2.0 / k0

        # init control loop integrators
        self.int_az = 0.0
        self.int_alt = 0.0

        self.rate_limit = rate_limit

    # returns new slew rates as a tuple (slew_az, slew_alt) in [phase units] 
    # per second, where [phase units] are the same as the units of the input
    # error values
    def update(self, error_az, error_alt):
        # proportional term
        prop_az = self.prop_gain * error_az
        prop_alt = self.prop_gain * error_alt

        # integral term
        self.int_az = clamp(self.int_az + self.int_gain * error_az, self.rate_limit)
        self.int_alt = clamp(self.int_alt + self.int_gain * error_alt, self.rate_limit)

        # output is the sum of proportional and integral terms subject to rate limit
        slew_rate_az = clamp(prop_az + self.int_az, self.rate_limit)
        slew_rate_alt = clamp(prop_alt + self.int_alt, self.rate_limit)
        return (slew_rate_az, slew_rate_alt)

class Tracker:

    def __init__(self, scope_device, observer, target):

        # define some limits
        self.slew_rate_limit = 16319.0 / 3600.0
        self.alt_min_limit = -10.0
        self.alt_max_limit = 67.0

        # update rate of control loop
        self.loop_period_s = 0.25

        self.loop_filter = LoopFilter(
            bandwidth = 0.5, 
            damping_factor = math.sqrt(2.0)/2.0, 
            update_period = self.loop_period_s, 
            rate_limit = self.slew_rate_limit
        )

        # Create connection to telescope
        self.scope = nexstar.NexStar(scope_device)

        # Create error source object
        self.error_source = BlindErrorSource(self.scope, observer, target)


    def start(self):
        self.start_time = time.time()
        self.running = True
        self.do_iteration()

    def stop(self):
        self.running = False

    def do_iteration(self):
        if self.running:
            threading.Timer(self.loop_period_s, self.do_iteration).start()
        else:
            return
        
        try:
            elapsed_time = time.time() - self.start_time

            # get current pointing error
            (error_az, error_alt) = self.error_source.compute_error()

            # loop filter -- outputs are new slew rates in degrees/second
            (slew_rate_az, slew_rate_alt) = self.loop_filter.update(error_az, error_alt)
            
            # enforce altitude limits
            (scope_az_deg, scope_alt_deg) = self.scope.get_azel()
            if scope_alt_deg >= self.alt_max_limit and slew_rate_alt > 0.0:
                slew_rate_alt = 0.0
                self.loop_filter.int_alt = 0.0
            elif scope_alt_deg <= self.alt_min_limit and slew_rate_alt < 0.0:
                slew_rate_alt = 0.0
                self.loop_filter.int_alt = 0.0 

            # update slew rates (arcseconds per second)
            self.scope.slew_var(slew_rate_az * 3600.0, slew_rate_alt * 3600.0)
        
        except:
            self.stop()
            raise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('tle', help='filename of two-line element (TLE) target ephemeris')
    parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
    args = parser.parse_args()

    # Create a PyEphem Observer object
    observer = ephem.Observer()
    observer.lat = args.lat
    observer.lon = args.lon
    observer.elevation = args.elevation

    # Create a PyEphem Body object corresonding to the TLE file
    tle = []
    with open(args.tle) as tlefile:
        for line in tlefile:
            tle.append(line)
    target = ephem.readtle(tle[0], tle[1], tle[2])

    tracker = Tracker(args.scope, observer, target)
    tracker.start()

    try:
        print('Press Enter to quit.')
        raw_input()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        print('Goodbye!')
