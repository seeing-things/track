#!/usr/bin/env python3

import time
import numpy as np
from astropy.io import fits
import cv2
import asi
import track

def asi_check(return_values):
    if isinstance(return_values, (tuple, list)):
        status_code = return_values[0]
        return_values = return_values[1:] if len(return_values) > 2 else return_values[1]
    else:
        status_code = return_values
        return_values = None

    if status_code != asi.ASI_SUCCESS:
        raise RuntimeError('return code: {}'.format(status_code))

    return return_values

def main():

    parser = track.ArgParser()
    parser.add_argument(
        '--exposure-time',
        help='camera exposure time in seconds',
        default=0.02,
        type=float
    )
    parser.add_argument(
        '--gain',
        help='camera gain',
        default=1,
        type=int
    )
    args = parser.parse_args()

    binning = 4

    if asi.ASIGetNumOfConnectedCameras() == 0:
        raise RuntimeError('No cameras connected')
    info = asi_check(asi.ASIGetCameraProperty(0))
    width = info.MaxWidth // binning
    height = info.MaxHeight // binning
    frame_size = width * height
    asi_check(asi.ASIOpenCamera(info.CameraID))
    asi_check(asi.ASIInitCamera(info.CameraID))
    asi_check(asi.ASISetROIFormat(
        info.CameraID,
        width,
        height,
        binning,
        asi.ASI_IMG_RAW8
    ))
    asi_check(asi.ASISetControlValue(
        info.CameraID,
        asi.ASI_EXPOSURE,
        int(args.exposure_time * 1e6),
        asi.ASI_FALSE
    ))
    asi_check(asi.ASISetControlValue(info.CameraID, asi.ASI_GAIN, args.gain, asi.ASI_FALSE))

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL);
    cv2.resizeWindow('camera', 640, 480);

    frame_count = 0
    while True:
        print('frame {:06d}'.format(frame_count))
        asi_check(asi.ASIStartExposure(info.CameraID, asi.ASI_FALSE))
        while True:
            status = asi_check(asi.ASIGetExpStatus(info.CameraID))
            if status == asi.ASI_EXP_SUCCESS:
                break
            elif status == asi.ASI_EXP_FAILED:
                raise RuntimeError('Exposure failed')
            else:
                time.sleep(0.01)

        frame = asi_check(asi.ASIGetDataAfterExp(info.CameraID, frame_size))
        frame = np.reshape(frame, (height, width))

        color_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        cv2.imshow('camera', color_frame)
        cv2.waitKey(1)
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        hdu = fits.PrimaryHDU(gray_frame)
        hdu.writeto('/tmp/test.fits', overwrite=True)

        subprocess.Run(
            [
                './bin/solve_field',
                '--overwrite',
                '--scale-low=2.2',
                '--scale-high=2.3',
                '/tmp/test.fits',
            ],
            cwd='/home/rgottula/dev/astrometry.net'
        )




if __name__ == "__main__":
    main()
