"""error sources for use in telescope tracking control loop.

The errorsources module provides classes that derive from the abstract ErrorSource class. Each
class defines a method compute_error() which returns an error vector representing the difference
between the current position and some measure of the ideal position. This information might come
from ephemeris data for a celestial or man-made object, a camera, or even human input. Error source
classes can also be designed to compute an error vector by combining data from multiple sources
or by intelligently switching between sources.
"""

from __future__ import print_function
import datetime
import math
from .control import ErrorSource
from .mathutils import *
import ephem
import numpy as np
try:
    import cv2
    from . import webcam
except ImportError as e:
    if 'cv2' in e.message:
        print('Failed to import cv2. Optical tracking requires OpenCV.')
        pass
    raise


class BlindErrorSource(ErrorSource):
    """Ephemeris based error source.

    This class implements an error source based on computed ephemeris information. It is dubbed
    a "blind" error source in the sense that no physical sensing of the target position is
    involved (such as by use of a camera). The error vector is computed by taking the difference
    of the mount's current position and the position of the target as predicted by the PyEphem
    package.

    Attributes:
        observer: PyEphem Observer object.
        target: PyEphem Target object.
        mount: TelescopeMount object.
        backlash_compensation: A boolean, True to enable backlash compensation.
        offset_callback: A function that can make adjustments to the target position.
        mount_position_cached: Cached position of the mount from last call to compute_error().
        target_position_cached: Cached position of the target from last call to compute_error().
    """

    def __init__(self, mount, observer, target, backlash_compensation=False):
        if backlash_compensation and not mount.backlash_supported():
            raise ValueError('mount does not support backlash compensation')
        self.observer = observer
        self.target = target
        self.mount = mount
        self.backlash_compensation = backlash_compensation
        self.offset_callback = None
        self.mount_position_cached = None
        self.target_position_cached = None
        self.axes = mount.get_axis_names()

    def register_offset_callback(self, callback):
        self.offset_callback = callback

    def get_axis_names(self):
        return self.axes

    def compute_error(self, retries=0):

        # Get coordinates of the target from a past time for use in determining direction of
        # motion. This needs to be far enough in the past that this position and the current
        # position are separated enough to compute an accurate motion vector.
        a_while_ago = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
        self.observer.date = ephem.Date(a_while_ago)
        self.target.compute(self.observer)
        target_position_prev = {}
        for axis in self.axes:
            target_position_prev[axis] = eval('self.target.' + axis) * 180.0 / math.pi

        # get coordinates of target for current time
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_position = {}
        for axis in self.axes:
            target_position[axis] = eval('self.target.' + axis) * 180.0 / math.pi
        self.target_position_cached = target_position

        # get current position of telescope (degrees)
        mount_position = self.mount.get_position()
        self.mount_position_cached = mount_position

        # make any corrections to predicted position
        if self.offset_callback is not None:
            adjusted_position = adjust_position(
                target_position_prev,
                target_position,
                self.offset_callback()
            )
        else:
            adjusted_position = target_position

        if self.backlash_compensation:
            target_motion_direction = {}
            for axis in self.axes:
                target_motion_direction[axis] = np.sign(
                    wrap_error(target_position[axis] - target_position_prev[axis])
                )

            # compensate for backlash if object is moving against the slew
            # direction used during alignment
            align_dir = self.mount.get_aligned_slew_dir()
            axes_to_adjust = {}
            for axis in self.axes:
                axes_to_adjust[axis] = align_dir[axis] != target_motion_direction[axis]
            mount_position = self.mount.remove_backlash(mount_position, axes_to_adjust)
            self.mount_position_cached = mount_position

        # compute pointing errors in degrees
        error = {}
        for axis in self.axes:
            error[axis] = wrap_error(adjusted_position[axis] - mount_position[axis])

        return error


class OpticalErrorSource(ErrorSource):

    def __init__(
            self,
            cam_dev_path,
            arcsecs_per_pixel,
            cam_num_buffers,
            cam_ctlval_exposure,
            x_axis_name,
            y_axis_name
        ):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0

        self.webcam = webcam.WebCam(cam_dev_path, cam_num_buffers, cam_ctlval_exposure)

        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

        self.frame_width_px = self.webcam.get_res_x()
        self.frame_height_px = self.webcam.get_res_y()

        self.frame_center_px = (self.frame_width_px / 2.0, self.frame_height_px / 2.0)

        # counts of consecutive frames with detections or no detections
        self.consec_detect_frames = 0
        self.consec_no_detect_frames = 0

        # detect OpenCV version to handle API differences between 2 and 3
        opencv_ver = int(cv2.__version__.split('.')[0])
        assert opencv_ver == 2 or opencv_ver == 3

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
        if opencv_ver == 2:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
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

    def get_axis_names(self):
        return [self.x_axis_name, self.y_axis_name]

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
            frame_annotated = cv2.drawKeypoints(
                frame,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.line(
                frame_annotated,
                (int(self.frame_center_px[0]), 0),
                (int(self.frame_center_px[0]), int(self.frame_height_px) - 1),
                (100, 0, 0),
                1
            )
            cv2.line(
                frame_annotated,
                (0, int(self.frame_center_px[1])),
                (int(self.frame_width_px) - 1, int(self.frame_center_px[1])),
                (100, 0, 0),
                1
            )
            cv2.imshow('frame', frame_annotated)
            cv2.waitKey(1)

            if not keypoints:
                if retries > 0:
                    retries -= 1
                    continue
                else:
                    self.consec_detect_frames = 0
                    self.consec_no_detect_frames += 1
                    raise self.NoSignalException('No target identified')

            # error is distance of first keypoint from center frame
            error_x_px = keypoints[0].pt[0] - self.frame_center_px[0]
            error_y_px = self.frame_center_px[1] - keypoints[0].pt[1]
            error_x_deg = error_x_px * self.degrees_per_pixel
            error_y_deg = error_y_px * self.degrees_per_pixel

            # FIXME: need to do proper coordinate transformation based on orientation of camera and
            # type of mount (az-alt versus equatorial)! Part of this fix should include finding
            # a better way of specifying the camera's orientation with respect to the mount axes
            # and eliminating the y_axis_name and x_axis_name constructor arguments and class
            # attributes.
            error = {}
            error[self.x_axis_name] = error_x_deg
            error[self.y_axis_name] = error_y_deg

            self.consec_detect_frames += 1
            self.consec_no_detect_frames = 0

            return error


class HybridErrorSource(ErrorSource):
    def __init__(
            self,
            mount,
            observer,
            target,
            cam_dev_path,
            arcsecs_per_pixel,
            cam_num_buffers,
            cam_ctlval_exposure,
            max_divergence=5.0,
            max_optical_no_signal_frames=4,
            backlash_compensation=False,
        ):
        self.axes = mount.get_axis_names()
        self.blind = BlindErrorSource(mount, observer, target, backlash_compensation)
        # FIXME: Have to do this because OpticalErrorSource has a crappy way of specifying how the
        # camera is oriented with respect to the mount axes.
        if set(self.axes) == set(['az', 'alt']):
            self.optical = OpticalErrorSource(
                cam_dev_path,
                arcsecs_per_pixel,
                cam_num_buffers,
                cam_ctlval_exposure,
                x_axis_name = 'az',
                y_axis_name = 'alt'
            )
        elif set(self.axes) == set(['ra', 'dec']):
            self.optical = OpticalErrorSource(
                cam_dev_path,
                arcsecs_per_pixel,
                cam_num_buffers,
                cam_ctlval_exposure,
                x_axis_name = 'ra',
                y_axis_name = 'dec'
            )
        else:
            raise ValueError('unrecognized axis names')
        self.max_divergence = max_divergence
        self.max_optical_no_signal_frames = max_optical_no_signal_frames
        self.state = 'blind'
        print('Hybrid error source starting in blind tracking state')

    def register_blind_offset_callback(self, callback):
        self.blind.register_offset_callback(callback)

    def compute_error(self, retries=0):
        blind_error = self.blind.compute_error(retries)

        try:
            optical_error = self.optical.compute_error(retries)
        except ErrorSource.NoSignalException:
            if self.state == 'blind':
                return blind_error
            else:
                if self.optical.consec_no_detect_frames >= self.max_optical_no_signal_frames:
                    print('Lost target in camera, switching to blind tracking')
                    self.state = 'blind'
                    return blind_error
                raise

        # get optical target position, which is the mount's current position plus the optical
        # error vector
        mount_position = self.blind.mount_position_cached
        target_position_optical = {}
        for axis in self.axes:
            target_position_optical[axis] = mount_position[axis] + optical_error[axis]

        # get blind object position, which is what PyEphem says it is
        target_position_blind = self.blind.target_position_cached

        # Get angle between optical and blind target position solutions. This is a measure of how
        # far the two solutions have diverged. A large divergence angle could mean that the
        # computer vision algorithm is not tracking the correct object.
        divergence_angle = angle_between(target_position_optical, target_position_blind)

        if self.state == 'blind' and divergence_angle < self.max_divergence:
            print('A target is in view, switching to optical tracking')
            self.state = 'optical'
        elif self.state == 'optical' and divergence_angle > self.max_divergence:
            print('Solutions diverged, switching to blind tracking')
            self.state = 'blind'

        return blind_error if self.state == 'blind' else optical_error
