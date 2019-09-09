#!/usr/bin/env python3

import os
import subprocess
import time
import tempfile
import numpy as np
from astropy.io import fits
from astropy import wcs
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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        # color_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        cv2.imshow('camera', gray_frame)
        cv2.waitKey(1)
        frame_count += 1

        with tempfile.TemporaryDirectory() as tempdir:

            filename_prefix = 'guidescope_frame'
            frame_filename = os.path.join(tempdir, filename_prefix + '.fits')

            hdu = fits.PrimaryHDU(gray_frame)
            hdu.writeto(frame_filename, overwrite=True)

            start_time = time.time()
            subprocess.run(
                [
                    './bin/solve-field',
                    '--overwrite',
                    '--scale-low=2.2',
                    '--scale-high=2.3',
                    frame_filename,
                ],
                cwd='/home/rgottula/dev/astrometry.net',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            elapsed = time.time() - start_time

            try:
                wcs_file = fits.open(os.path.join(tempdir, filename_prefix + '.wcs'))
            except FileNotFoundError as e:
                print('no solution found :(')
                continue

            print('Solution found in {} seconds'.format(elapsed))

            wcs_header = wcs.WCS(header=wcs_file[0].header)

            # get "world coordinates" for center of frame
            ra, dec = wcs_header.wcs_pix2world((width - 1) / 2.0, (height - 1) / 2.0, 0)

            print('Found solution at ({}, {})'.format(ra, dec))


if __name__ == "__main__":
    main()
