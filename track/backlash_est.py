#!/usr/bin/env python3

"""Backlash estimation for telescope mount.

This program is a tool to help estimate the magnitudes of backlash deadbands in a motorized
telescope mount. This is accomplished by using computer vision through a camera to track features
while the mount is moved back and forth in a rectangular pattern. The dimensions of the rectangle
are controlled in each axis by the user with the W, A, S, and D keys. By experimentation, the
dimensions of this movement pattern are set to the maximum size possible without discernable
movement of features in the camera. The resulting values are the estimates of the backlash
deadbands.

Currently hard-coded for Az-Alt axis names. Could be refactored to support equatorial mounts.

OpenCV code for optical flow is adapted from this example:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
"""

from __future__ import print_function
import sys
import numpy as np
import cv2
import track

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--camera',
        help='device node path for tracking webcam',
        default='/dev/video0'
    )
    parser.add_argument(
        '--camera-bufs',
        help='number of webcam capture buffers',
        required=True,
        type=int
    )
    parser.add_argument(
        '--camera-exposure',
        help='webcam exposure level',
        default=3200,
        type=int
    )
    parser.add_argument(
        '--mount-type',
        help='select mount type (nexstar or gemini)',
        default='gemini'
    )
    parser.add_argument(
        '--mount-path',
        help='serial device node or hostname for mount command interface',
        default='/dev/ttyACM0'
    )
    parser.add_argument(
        '--bypass-alt-limits',
        help='bypass altitude limit enforcement',
        action='store_true'
    )
    args = parser.parse_args()

    if args.mount_type == 'nexstar':
        mount = track.NexStarMount(args.mount_path, bypass_alt_limits=args.bypass_alt_limits)
    elif args.mount_type == 'gemini':
        mount = track.LosmandyGeminiMount(args.mount_path)
    else:
        print('mount-type not supported: ' + args.mount_type)
        sys.exit(1)
    position_start = mount.get_position()
    deadband_az = 100.0
    deadband_alt = 100.0
    slew_rate = 100.0 / 3600.0

    mount.slew('az', slew_rate)
    mount.slew('alt', 0.0)
    direction = 'right'

    webcam = track.WebCam(args.camera, args.camera_bufs, args.camera_exposure)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=15)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    print('searching for candidate set of features to track')
    while True:
        old_frame = webcam.get_fresh_frame()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        cv2.imshow('frame', old_frame)
        cv2.waitKey(30)
        if p0 is not None:
            break

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        frame = webcam.get_fresh_frame()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.putText(
            img,
            "az deadband: " + str(deadband_az),
            (0, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255)
        )
        cv2.putText(
            img,
            "alt deadband: " + str(deadband_alt),
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255)
        )
        cv2.putText(
            img,
            "slew rate: " + str(slew_rate * 3600.0),
            (0, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255)
        )

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('c'):
            mask = np.zeros_like(old_frame)
        elif k == ord('r'):
            ptmp = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if ptmp is not None:
                p0 = ptmp
                mask = np.zeros_like(old_frame)
                old_gray = frame_gray.copy()
                continue
            else:
                print('Could not find any new features to track')
        elif k == ord('w'):
            deadband_alt += 1
        elif k == ord('s'):
            deadband_alt -= 1
        elif k == ord('a'):
            deadband_az -= 1
        elif k == ord('d'):
            deadband_az += 1
        elif k == ord('='):
            slew_rate += 10.0 / 3600.0
        elif k == ord('-'):
            slew_rate -= 10.0 / 3600.0

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        position = mount.get_position()
        position_change = {
            'az': track.wrap_error(position['az'] - position_start['az']) * 3600.0,
            'alt': track.wrap_error(position['alt'] - position_start['alt']) * 3600.0,
        }
        print(str(position_change))
        if direction == 'right' and position_change['az'] > deadband_az:
            mount.slew('az', 0.0)
            mount.slew('alt', +slew_rate)
            direction = 'up'
        elif direction == 'up' and position_change['alt'] > deadband_alt:
            mount.slew('alt', 0.0)
            mount.slew('az', -slew_rate)
            direction = 'left'
        elif direction == 'left' and position_change['az'] <= 0.0:
            mount.slew('az', 0.0)
            mount.slew('alt', -slew_rate)
            direction = 'down'
        elif direction == 'down' and position_change['alt'] <= 0.0:
            mount.slew('alt', 0.0)
            mount.slew('az', +slew_rate)
            direction = 'right'

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
