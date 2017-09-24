#!/usr/bin/env python


import random
import time

from webcam import WebCam


random.seed()

cam = WebCam('/dev/video0', 15, 1500)

num = 0
slept = 0
while True:
    print('====== {:4d} | slept: {:3d}ms ======'.format(num, slept))
    frame = cam.get_fresh_frame()
    num += 1
    slept = max(1, min(500, int(random.gammavariate(1.0, 120.0))))
    time.sleep(slept / 1000.0)
