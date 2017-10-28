#!/usr/bin/env python

import config
import configargparse
import mounts
import time
from errorsources import wrap_error
import matplotlib.pyplot as plt
import numpy as np

parser = configargparse.ArgParser(default_config_files=config.DEFAULT_FILES)
parser.add_argument('--scope', help='serial device for connection to telescope', default='/dev/ttyUSB0')
parser.add_argument('--bypass-alt-limits', help='bypass mount altitude limits', action='store_true')
args = parser.parse_args()

mount = mounts.NexStarMount(args.scope, bypass_alt_limits=args.bypass_alt_limits)
if args.bypass_alt_limits:
    print('Warning: Altitude limits disabled! Be careful!')

t = []
p = []
try:
    position_start = mount.get_azalt()['az']
    time_start = time.time()
    time_elapsed = 0
    while time_elapsed < 1.0:
        time_elapsed = time.time() - time_start
        t.append(time_elapsed)
        p.append(wrap_error(mount.get_azalt()['az'] - position_start))
    mount.slew('az', -4.0)
    while time_elapsed < 10.0:
        time_elapsed = time.time() - time_start
        t.append(time_elapsed)
        p.append(wrap_error(mount.get_azalt()['az'] - position_start))
    mount.slew('az', 0.0)

    t = np.asarray(t, dtype='float')
    p = np.asarray(p, dtype='float')

    plt.plot(t, p)
    plt.show()

    tdiff = np.diff(t)
    pdiff = np.diff(p)

    # estimate slew rate as a function of time by taking first difference of
    # position samples
    sr = pdiff / tdiff
    # sr2 = pdiff / np.mean(tdiff)

    # differentiate slew rate to get estimate of impulse response
    h = np.diff(sr)

    # plt.plot(t[:-1], sr, '.-', t[:-1], sr2, '.-')
    plt.plot(t[:-1], sr, '.-')
    plt.show()

    plt.plot(t[:-2], h, '.-')
    plt.show()

except KeyboardInterrupt:
    pass
