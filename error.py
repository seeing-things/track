from track import ErrorSource
import ephem
import datetime
import math
import cv2
import numpy as np
import exceptions

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


class OpticalErrorSource(ErrorSource):

    def __init__(self, device_name, arcsecs_per_pixel):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0
        self.camera = cv2.VideoCapture(device_name)

        if not self.camera.isOpened():
            raise exceptions.IOError('Could not open camera')

        frame_width = self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        frame_height = self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.frame_center = (frame_width / 2.0, frame_height / 2.0)
        
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
        self.detector = cv2.SimpleBlobDetector(params)

        cv2.namedWindow('frame')
        cv2.createTrackbar('block size', 'frame', 7, 31, self.block_size_validate)
        cv2.createTrackbar('C', 'frame', 3, 255, self.do_nothing)

    # validator for block size trackbar
    def block_size_validate(x):
        if x % 2 == 0:
            cv2.setTrackbarPos('block size', 'frame', x + 1)
        elif x < 3:
            cv2.setTrackbarPos('block size', 'frame', 3)

    # validator for OpenCV trackbar
    def do_nothing(x):
        pass

    def compute_error(self):

        ret, frame = self.camera.read()

        if not ret:
            raise exceptions.IOError('Could not get frame from camera')

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

        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the resulting frame
        cv2.imshow('frame', frame_with_keypoints)
        cv2.waitKey(1)

        print('num keypoints: ' + str(len(keypoints)))
        if not keypoints:
            raise self.NoSignalException('No target identified')

        # error is distance of first keypoint from center frame
        error_x_px = keypoints[0].pt[0] - self.frame_center[0]
        error_y_px = self.frame_center[1] - keypoints[0].pt[1]
        error_x_deg = error_x_px * self.degrees_per_pixel
        error_y_deg = error_y_px * self.degrees_per_pixel

        # FIXME: need to do proper coordinate transformation based on orientation of camera!
        error_az = error_x_deg
        error_alt = error_y_deg

        print('error: ' + str((error_az, error_alt)))

        return (error_az, error_alt)
