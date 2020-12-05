#!/usr/bin/env python3

"""program for testing plate solving with guidescope camera"""

import time
from datetime import datetime
import numpy as np
import asi
import cv2
import track
from track import cameras

def main():
    """Repeatedly prints coordinates of camera frame center found by plate solving."""

    parser = track.ArgParser()
    parser.add_argument(
        '--skip-solve',
        help='skip plate solving',
        action='store_true'
    )
    cameras.add_program_arguments(parser, profile='align')
    args = parser.parse_args()

    camera = cameras.make_camera_from_args(args, profile='align')

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('camera', 640, 480)

    frame_count = 0
    while True:
        print(f'Frame {frame_count:06d} at {datetime.now()}...', end='', flush=True)
        frame_count += 1
        frame = camera.get_frame()
        cv2.imshow('camera', frame)
        cv2.waitKey(10)

        if args.skip_solve:
            continue

        try:
            start_time = time.time()
            sc = track.plate_solve(
                frame,
                camera_width=camera.field_of_view[1]
            )
            elapsed = time.time() - start_time
        except track.NoSolutionException:
            print('No solution found')
            continue

        print(f'Found solution at ({sc.to_string("decimal")}) in {elapsed} seconds')


if __name__ == "__main__":
    main()
