#!/usr/bin/env python

import sys
import ephem
import time
import datetime
import math
import urllib2
import nexstar
import threading
import multiprocessing
import matplotlib.pyplot as plt
import argparse

def clamp(x, limit):
    return max(min(limit, x), -limit)

def wrap_error(e):
    return (e + 180.0) % 360.0 - 180.0

def plot_process(q):
    
    # open a plot figure and set it up
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.axis([0, 100, 0, 100])
    ax.plot([0, 100], [0, 100], 'k-')
    line1, = ax.plot([0], [0], 'r-') # Returns a tuple of line targetects, thus the comma
    plt.grid()
    ax2 = fig.add_subplot(212)
    plt.axis([0, 100, -10.0, 10.0])
    ax2.plot([0, 100], [0, 0], 'k-')
    line_error_az, = ax2.plot([0], [0], 'r-')
    line_error_alt, = ax2.plot([0], [0], 'b-')
    plt.grid()

    while True:
        result = q.get()
        line_error_az.set_xdata(result[0])
        line_error_az.set_ydata(result[1])
        line_error_alt.set_xdata(result[0])
        line_error_alt.set_ydata(result[2])
        fig.canvas.draw()


class Tracker:

    def __init__(self, scope_device, observer, target):

        # define some limits
        self.slew_limit = 16319.0 / 3600.0
        self.alt_min_limit = -10.0
        self.alt_max_limit = 67.0

        # loop filter parameters
        self.loop_bw_hz = 0.5
        self.loop_zeta = math.sqrt(2.0) / 2.0
        self.loop_period_s = 0.25

        # compute the loop filter gains
        loop_bt = self.loop_bw_hz * self.loop_period_s
        k0 = self.loop_period_s
        denom = self.loop_zeta + 1.0 / (4.0 * self.loop_zeta)
        self.prop_gain = 4.0 * self.loop_zeta / denom * loop_bt / k0
        self.int_gain = 4.0 / denom**2.0 * loop_bt**2.0 / k0 

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # Create connection to telescope
        self.scope = nexstar.NexStar(scope_device)

        # init control loop integrators
        self.int_az = 0.0
        self.int_alt = 0.0

        # data for debugging
        self.time_list = []
        self.scope_az_list = []
        self.scope_alt_list = []
        self.error_az_list = []
        self.error_alt_list = []


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
            self.time_list.append(elapsed_time)

            # get current coordinates of the target
            # not using ephem.now() because it rounds time to the nearest second
            self.observer.date = ephem.Date(datetime.datetime.utcnow())
            self.target.compute(self.observer)
            #print('Az: %s Alt: %s' % (target.az, target.alt))
            target_az_deg = self.target.az * 180.0 / math.pi
            target_alt_deg = self.target.alt * 180.0 / math.pi
            #print('target: ' + str(target_az_deg) + ', ' + str(target_alt_deg))

            # get current position of telescope (degrees)
            (scope_az_deg, scope_alt_deg) = self.scope.get_azel()
            print('scope: ' + str(scope_az_deg) + ', ' + str(scope_alt_deg))
            self.scope_az_list.append(scope_az_deg)
            self.scope_alt_list.append(scope_alt_deg)
             
            # compute pointing errors in degrees
            error_az = wrap_error(target_az_deg - scope_az_deg)
            error_alt = wrap_error(target_alt_deg - scope_alt_deg)
            #self.error_az_list.append(error_az)
            #self.error_alt_list.append(error_alt)
            # JUSTIN TEMP: use arcseconds rather than degrees
            self.error_az_list.append(error_az * 60.0 * 60.0)
            self.error_alt_list.append(error_alt * 60.0 * 60.0)

            #print('' + str(elapsed_time) + ', ' + str(error_az) + ', ' + str(error_alt))

            # loop filters -- outputs are new slew rates in degrees/second
            prop_az = self.prop_gain * error_az
            prop_alt = self.prop_gain * error_alt
            self.int_az = clamp(self.int_az + self.int_gain * error_az, self.slew_limit)
            self.int_alt = clamp(self.int_alt + self.int_gain * error_alt, self.slew_limit)
            slew_az = clamp(prop_az + self.int_az, self.slew_limit)
            slew_alt = clamp(prop_alt + self.int_alt, self.slew_limit)
            
            # enforce altitude limits
            if scope_alt_deg >= self.alt_max_limit and slew_alt > 0.0:
                slew_alt = 0.0
                self.int_alt = 0.0
            elif scope_alt_deg <= self.alt_min_limit and slew_alt < 0.0:
                slew_alt = 0.0
                self.int_alt = 0.0 

            # update slew rates (arcseconds per second)
            self.scope.slew_var(slew_az * 3600.0, slew_alt * 3600.0)
        
        except:
            self.stop()
            raise

        #q.put((self.time_list, self.error_az_list, self.error_alt_list))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('tle', help='filename of two-line element (TLE) target ephemeris')
    parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
    args = parser.parse_args()

    #global q
    #q = multiprocessing.Queue()

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
        tracker.stop()
        print('Goodbye!')
    except KeyboardInterrupt:
        tracker.stop()
        print('Goodbye!')

    #proc = multiprocessing.Process(None, plot_process, args=(q,))
    #proc.start()

    #while True:
    #    time.sleep(0.5)
