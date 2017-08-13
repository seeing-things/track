from track import ErrorSource
import ephem
import datetime
import math
import cv2
import numpy as np
import time
import select
import v4l2capture
from PIL import Image

# wraps an angle in degrees to the range [-180,+180)
def wrap_error(e):
    return (e + 180.0) % 360.0 - 180.0

class BlindErrorSource(ErrorSource):

    def __init__(self, mount, observer, target):

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # TelescopeMount object
        self.mount = mount

    def compute_error(self, retries=0):

        # get current coordinates of the target in degrees
        # not using ephem.now() because it rounds time to the nearest second
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_az_deg = self.target.az * 180.0 / math.pi
        target_alt_deg = self.target.alt * 180.0 / math.pi

        # get current position of telescope (degrees)
        mount_position_deg = self.mount.get_azalt()
         
        # compute pointing errors in degrees
        error = {}
        error['az'] = wrap_error(target_az_deg - mount_position_deg['az'])
        error['alt'] = wrap_error(target_alt_deg - mount_position_deg['alt'])

        return error


class OpticalErrorSource(ErrorSource):

    def __init__(self, device_name, arcsecs_per_pixel, wanted_width_px, wanted_height_px, num_buffers):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0

        self.buf_max = num_buffers
        self.buf_cur = 0

        # try to set resolution to the desired W/H; also request RGB, not YUV
        self.camera = v4l2capture.Video_device(device_name)
        self.frame_width_px, self.frame_height_px = self.camera.set_format(wanted_width_px, wanted_height_px, 0)

        # set up the camera for capturing frames
        self.camera.create_buffers(num_buffers)
        self.camera.queue_all_buffers()
        self.camera.start()


        self.frame_center_px = (self.frame_width_px / 2.0, self.frame_height_px / 2.0)

        # initialize blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.maxArea = 50000.0
        #params.thresholdStep = 1
        params.minThreshold = 100
        params.maxThreshold = 200
        params.minDistBetweenBlobs = 200
        self.detector = cv2.SimpleBlobDetector_create(params)

        cv2.namedWindow('frame')
        cv2.createTrackbar('block size', 'frame', 7, 31, self.block_size_validate)
        cv2.createTrackbar('C', 'frame', 3, 255, self.do_nothing)

    def __del__(self):
        # clean up the camera frame capture resources
        self.camera.stop()
        self.camera.close()

    # validator for block size trackbar
    def block_size_validate(self, x):
        if x % 2 == 0:
            cv2.setTrackbarPos('block size', 'frame', x + 1)
        elif x < 3:
            cv2.setTrackbarPos('block size', 'frame', 3)

    # validator for OpenCV trackbar
    def do_nothing(self, x):
        pass

    def compute_error(self, retries=0):

        while True:
            # wait for the camera to have a frame ready
            select.select((self.camera,), (), ())

            frame_raw = self.camera.read()
            frame_img = Image.frombytes('RGB', (self.frame_width_px, self.frame_height_px), frame_raw)
            frame = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)

            # keep track of the capture buffer state and requeue if neceeesary
            self.buf_cur += 1
            if self.buf_cur >= self.buf_max:
                self.camera.queue_all_buffers()
                self.buf_cur = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                cv2.getTrackbarPos('block size', 'frame'),
                cv2.getTrackbarPos('C', 'frame')
            )

            keypoints = self.detector.detect(thresh)

            # display the original frame with keypoints circled in red
            frame_annotated = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.line(frame_annotated, (int(self.frame_center_px[0]), 0), (int(self.frame_center_px[0]), int(self.frame_height_px) - 1), (100,0,0), 1)
            cv2.line(frame_annotated, (0, int(self.frame_center_px[1])), (int(self.frame_width_px) - 1, int(self.frame_center_px[1])), (100,0,0), 1)
            cv2.imshow('frame', frame_annotated)
            cv2.waitKey(1)

            if not keypoints:
                if retries > 0:
                    retries -= 1
                    continue
                else:
                    raise self.NoSignalException('No target identified')

            # error is distance of first keypoint from center frame
            error_x_px = keypoints[0].pt[0] - self.frame_center_px[0]
            error_y_px = self.frame_center_px[1] - keypoints[0].pt[1]
            error_x_deg = error_x_px * self.degrees_per_pixel
            error_y_deg = error_y_px * self.degrees_per_pixel

            # FIXME: need to do proper coordinate transformation based on orientation of camera!
            error = {}
            error['az'] = error_x_deg
            error['alt'] = error_y_deg

            return error
