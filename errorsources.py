from track import ErrorSource
import ephem
import datetime
import math
import cv2
import numpy as np
import webcam

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
        target_position_prev = {
            'az': self.target.az * 180.0 / math.pi,
            'alt': self.target.alt * 180.0 / math.pi
        }

        # get current position of telescope (degrees)
        mount_position = self.mount.get_azalt()

        # get coordinates of target a second time to determine direction of motion
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_position = {
            'az': self.target.az * 180.0 / math.pi,
            'alt': self.target.alt * 180.0 / math.pi
        }

        target_motion_direction = {
            'az': np.sign(wrap_error(target_position['az'] - target_position_prev['az'])),
            'alt': np.sign(wrap_error(target_position['alt'] - target_position_prev['alt'])),
        }

        # compensate for backlash if object is moving against the slew 
        # direction used during alignment
        align_dir = mount.get_align_dir()
        axes_to_adjust = {
            'az': align_dir['az'] != target_motion_direction['az'],
            'alt': align_dir['alt'] != target_motion_direction['alt'],
        }
        mount_position = mount.remove_backlash(mount_position, axes_to_adjust)
         
        # compute pointing errors in degrees
        error = {}
        error['az'] = wrap_error(target_position['az'] - mount_position['az'])
        error['alt'] = wrap_error(target_position['alt'] - mount_position['alt'])

        return error


class OpticalErrorSource(ErrorSource):

    def __init__(self, cam_dev_path, arcsecs_per_pixel, cam_res_wanted, cam_num_buffers):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0

        self.webcam = webcam.WebCam(cam_dev_path, cam_res_wanted, cam_num_buffers)

        self.frame_width_px  = self.webcam.get_res_x()
        self.frame_height_px = self.webcam.get_res_y()

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
            frame = self.webcam.get_fresh_frame()
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


class HybridErrorSource(ErrorSource):
    def __init__(
            self,
            mount,
            observer,
            target,
            cam_dev_path,
            arcsecs_per_pixel,
            cam_res_wanted,
            cam_num_buffers,
            max_divergence=2.0
        ):
        self.blind = BlindErrorSource(mount, observer, target)
        self.optical = OpticalErrorSource(cam_dev_path, arcsecs_per_pixel, cam_res_wanted, cam_num_buffers)
        self.max_divergence = max_divergence

        self.BLIND_STATE = 0
        self.OPTICAL_STATE = 1
        self.state = self.BLIND_STATE

        print('Hybrid error source starting in blind tracking state')

    def compute_error(self):
        blind_error = self.blind.compute_error()

        try:
            optical_error = self.optical.compute_error()
        except ErrorSource.NoSignalException:
            if self.state == self.BLIND_STATE:
                return blind_error
            else:
                raise

        error_diff = {
            'az': blind_error['az'] - optical_error['az'],
            'alt': blind_error['alt'] - optical_error['alt'],
        }
        error_diff_mag = np.absolute(error_diff['az'] + 1j*error_diff['alt'])
        if self.state == self.BLIND_STATE and error_diff_mag < self.max_divergence:
            print('Solutions within ' + str(error_diff_mag) + ', switched to optical tracking')
            self.state = self.OPTICAL_STATE
        elif self.state == self.OPTICAL_STATE and error_diff_mag >= self.max_divergence:
            print('Solutions diverged (' + str(error_diff_mag) + ' degrees apart), switching back to blind tracking')
            self.state = self.BLIND_STATE

        return blind_error if self.state == self.BLIND_STATE else optical_error
