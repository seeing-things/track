#!/usr/bin/env python3

"""program for testing plate solving with guidescope camera"""

import time
import numpy as np
import asi
import cv2
import track

def main():
    """Repeatedly prints coordinates of camera frame center found by plate solving."""

    parser = track.ArgParser()
    parser.add_argument(
        '--camera-res',
        help='guidescope camera resolution in arcseconds per pixel',
        required=True,
        type=float
    )
    parser.add_argument(
        '--exposure-time',
        help='camera exposure time in seconds',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--gain',
        help='camera gain',
        default=400,
        type=int
    )
    parser.add_argument(
        '--binning',
        help='camera binning',
        default=4,
        type=int
    )
    parser.add_argument(
        '--skip-solve',
        help='skip plate solving',
        action='store_true'
    )
    args = parser.parse_args()

    if asi.ASIGetNumOfConnectedCameras() == 0:
        raise RuntimeError('No cameras connected')
    camera_info = asi.ASICheck(asi.ASIGetCameraProperty(0))
    width = camera_info.MaxWidth // args.binning
    height = camera_info.MaxHeight // args.binning
    frame_size = width * height * 2
    asi.ASICheck(asi.ASIOpenCamera(camera_info.CameraID))
    asi.ASICheck(asi.ASIInitCamera(camera_info.CameraID))
    asi.ASICheck(asi.ASISetROIFormat(
        camera_info.CameraID,
        width,
        height,
        args.binning,
        asi.ASI_IMG_RAW16
    ))
    asi.ASICheck(asi.ASISetControlValue(
        camera_info.CameraID,
        asi.ASI_EXPOSURE,
        int(args.exposure_time * 1e6),
        asi.ASI_FALSE
    ))
    asi.ASICheck(asi.ASISetControlValue(
        camera_info.CameraID,
        asi.ASI_GAIN,
        args.gain,
        asi.ASI_FALSE
    ))
    asi.ASICheck(asi.ASISetControlValue(
        camera_info.CameraID,
        asi.ASI_MONO_BIN,
        1,
        asi.ASI_FALSE
    ))

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('camera', 640, 480)

    frame_count = 0
    while True:
        print('frame {:06d}'.format(frame_count))
        frame_count += 1

        asi.ASICheck(asi.ASIStartExposure(camera_info.CameraID, asi.ASI_FALSE))
        while True:
            status = asi.ASICheck(asi.ASIGetExpStatus(camera_info.CameraID))
            if status == asi.ASI_EXP_SUCCESS:
                break
            if status == asi.ASI_EXP_FAILED:
                raise RuntimeError('Exposure failed')
            time.sleep(0.01)
        frame = asi.ASICheck(asi.ASIGetDataAfterExp(camera_info.CameraID, frame_size))
        frame = frame.view(dtype=np.uint16)
        frame = np.reshape(frame, (height, width))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        cv2.imshow('camera', gray_frame)
        cv2.waitKey(10)

        if args.skip_solve:
            continue

        try:
            start_time = time.time()
            sc = track.plate_solve(
                gray_frame,
                camera_width=(camera_info.MaxWidth * args.camera_res / 3600.0)
            )
            elapsed = time.time() - start_time
        except track.NoSolutionException:
            print('No solution found')
            continue

        print('Found solution at ({}) in {} seconds'.format(sc.to_string('decimal'), elapsed))


if __name__ == "__main__":
    main()
