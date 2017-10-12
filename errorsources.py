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

def normalize(v):
    """Normalize a Cartesian vector in 3-space to unit magnitude.

    Args:
        v: A vector in 3-space using Cartesian coordinates.

    Returns:
        A vector in 3-space with the same direction as v but with unit
            magnitude.
    """
    v = np.asarray(v)
    return v / math.sqrt(np.dot(v, v))

def rotate(v, axis, theta):
    """Rotates a vector in 3-space.

    Rotate a vector v in 3-space about the given axis of rotation by theta
    in a counterclockwise direction (right-hand rule). The rotation matrix is
    formed using the Euler-Rodrigues formula.

    Reference: https://stackoverflow.com/a/6802723

    Args:
        v: A vector in 3-space using Cartesian coordinates.
        axis: A vector in 3-space giving the axis of rotation. Not required to
            be a unit vector (magnitude ignored).
        theta: Angle of rotation in degrees. The rotation will be performed
            according to the right-hand rule convention.

    Returns:
        A numpy array with size 3 representing the rotated vector.
    """

    theta = theta * math.pi / 180.0

    # make the rotation matrix
    axis = np.asarray(axis)
    axis = normalize(axis)
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rotation_matrix =  np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return np.dot(rotation_matrix, v)

def horiz_to_cart(v):
    """Convert unit vector from horizontal to Cartesian coordinate system.

    Converts a unit vector in 3-space from the horizontal (Azimuth-Altitude)
    coordinate system to the Cartesian (x,y,z) coordinate system. The
    horizontal coordinate system is similar to the spherical coordinate system
    except (1) the azimuthal angle increases in a clockwise direction about
    the z-axis, and (2) the altitude is zero at the azimuthal (xy) plane, in
    contrast to the polar angle of the spherical coordinate system which is
    zero at zenith.

    Args:
        v: A dict containing keys 'az' and 'alt' with values in degrees.
            It is assumed that the magnitude of this vector is 1.

    Returns:
        A size 3 numpy array containing the Cartesian coordinates of the vector.
    """

    # degrees to radians
    az = v['az'] * math.pi / 180.0
    alt = v['alt'] * math.pi / 180.0

    # horizontal to spherical
    r = 1
    phi = -az
    theta = math.pi / 2.0 - alt

    # spherical to Cartesian
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return np.asarray([x,y,z], dtype=float)

def cart_to_horiz(v):
    """Convert a vector from Cartesian to horizontal coordinate system.

    Converts a vector in 3-space from the Cartesian (x,y,z) coordinate
    system to the horizontal (Azimuth-Altitude) coordinate system. Any
    magnitude information will be lost. See the horiz_to_cart description for
    differences between horizontal and spherical coordinate systems.

    Args:
        v: A size 3 numpy array containing the Cartesian coordinates of a
            vector.

    Returns:
        A dict containing keys 'az' and 'alt' with values in degrees.
    """
    x = v[0]
    y = v[1]
    z = v[2]

    # Cartesian to spherical (unit magnitude assumed)
    theta = math.acos(z)
    phi = math.atan2(y, x)

    # spherical to horizontal
    az = (-phi) % (2.0*math.pi)
    alt = math.pi / 2.0 - theta

    # radians to degrees
    az_deg = 180.0 / math.pi * az
    alt_deg = 180.0 / math.pi * alt

    return {'az': az_deg, 'alt': alt_deg}

def adjust_position(target_position_prev, target_position, offset):
    """Adjust target position by correction factor.

    Adjusts the position of the target by an offset where the offset is
    specified in a reference frame defined by the object's direction of
    travel. This makes it possible to make adjustments such as "1 degree
    ahead of the predicted position" or "0.3 degrees left with respect to
    the object's direction of travel."  This is expected to be more useful
    than adjustments such as "1 degree higher in altitude."

    Args:
        target_position_prev: A dict with keys 'az' and 'alt' giving the
            position of the target a short time ago in degrees.
        target_position: A dict with keys 'az' and 'alt' giving the current
            position of the target in degrees.
        offset: A two-element list or numpy array giving the offset adjustments
            in degrees. The first element is the x-axis offset and the second
            element is the y-axis offset. The +y axis is in the direction of
            the object's motion. The x-axis is perpendicular to the object's
            motion.

    Returns:
        A dict with keys 'az' and 'alt' giving the adjusted position.
    """
    # convert vectors in horizontal coordinate system to Cartesian
    tpos_prev = horiz_to_cart(target_position_prev)
    tpos = horiz_to_cart(target_position)

    if all(tpos == tpos_prev):
        raise ValueError('current and previous positions are equal!')

    # Compute object motion vector. This makes the assumption that tpos and
    # tpos_prev are separated by a small angle. The more correct calculation
    # would result in a vector tmotion that is tangent to the unit sphere at
    # the location of tpos.
    tmotion = normalize(tpos - tpos_prev)

    # convert gamepad vector from Cartesian to polar
    gpcomplex = offset[0] + 1j*offset[1]
    gpmag = np.abs(gpcomplex)
    gparg = np.angle(gpcomplex) * 180.0 / math.pi

    # compute axis of rotation
    axis = rotate(tmotion, tpos, gparg)

    # rotate position about axis of rotation
    tpos_new = rotate(tpos, axis, gpmag)

    # convert back to horizontal coordinate system
    return cart_to_horiz(tpos_new)


class BlindErrorSource(ErrorSource):

    def __init__(self, mount, observer, target):

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # TelescopeMount object
        self.mount = mount

        self.offset_callback = None

    def register_offset_callback(self, callback):
        self.offset_callback = callback

    def compute_error(self, retries=0):

        # Get coordinates of the target from a past time for use in determining
        # direction of motion. This needs to be far enough in the past that
        # this position and the current position are separated enough to
        # compute an accurate motion vector.
        a_while_ago = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
        self.observer.date = ephem.Date(a_while_ago)
        self.target.compute(self.observer)
        target_position_prev = {
            'az': self.target.az * 180.0 / math.pi,
            'alt': self.target.alt * 180.0 / math.pi
        }

        # get coordinates of target for current time
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_position = {
            'az': self.target.az * 180.0 / math.pi,
            'alt': self.target.alt * 180.0 / math.pi
        }

        # get current position of telescope (degrees)
        mount_position = self.mount.get_azalt()

        # make any corrections to predicted position
        if self.offset_callback is not None:
            adjusted_position = adjust_position(
                target_position_prev,
                target_position,
                self.offset_callback()
            )
        else:
            adjusted_position = target_position

        target_motion_direction = {
            'az': np.sign(wrap_error(target_position['az'] - target_position_prev['az'])),
            'alt': np.sign(wrap_error(target_position['alt'] - target_position_prev['alt'])),
        }

        # compensate for backlash if object is moving against the slew
        # direction used during alignment
        align_dir = self.mount.get_aligned_slew_dir()
        axes_to_adjust = {
            'az': align_dir['az'] != target_motion_direction['az'],
            'alt': align_dir['alt'] != target_motion_direction['alt'],
        }
        mount_position = self.mount.remove_backlash(mount_position, axes_to_adjust)

        # compute pointing errors in degrees
        error = {}
        error['az'] = wrap_error(adjusted_position['az'] - mount_position['az'])
        error['alt'] = wrap_error(adjusted_position['alt'] - mount_position['alt'])

        return error


class OpticalErrorSource(ErrorSource):

    def __init__(self, cam_dev_path, arcsecs_per_pixel, cam_num_buffers, cam_ctlval_exposure):

        self.degrees_per_pixel = arcsecs_per_pixel / 3600.0

        self.webcam = webcam.WebCam(cam_dev_path, cam_num_buffers, cam_ctlval_exposure)

        self.frame_width_px  = self.webcam.get_res_x()
        self.frame_height_px = self.webcam.get_res_y()

        self.frame_center_px = (self.frame_width_px / 2.0, self.frame_height_px / 2.0)

        # counts of consecutive frames with detections or no detections
        self.consec_detect_frames = 0
        self.consec_no_detect_frames = 0

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
                    self.consec_detect_frames = 0
                    self.consec_no_detect_frames += 1
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
            max_optical_no_signal_frames=4
        ):
        self.blind = BlindErrorSource(mount, observer, target)
        self.optical = OpticalErrorSource(cam_dev_path, arcsecs_per_pixel, cam_num_buffers, cam_ctlval_exposure)
        self.max_divergence = max_divergence
        self.max_optical_no_signal_frames = max_optical_no_signal_frames
        self.state = 'blind'
        print('Hybrid error source starting in blind tracking state')

    def register_blind_offset_callback(self, callback):
        self.blind.register_offset_callback(callback)

    def compute_error(self):
        blind_error = self.blind.compute_error()

        try:
            optical_error = self.optical.compute_error()
        except ErrorSource.NoSignalException:
            if self.state == 'blind':
                return blind_error
            else:
                if self.optical.consec_no_detect_frames >= self.max_optical_no_signal_frames:
                    print('Lost target in camera, switching to blind tracking')
                    self.state = 'blind'
                    return blind_error
                raise

        # this is not the right way to compute the angle between the two positions!
        error_diff = {
            'az': blind_error['az'] - optical_error['az'],
            'alt': blind_error['alt'] - optical_error['alt'],
        }
        error_diff_mag = np.absolute(error_diff['az'] + 1j*error_diff['alt'])
        if self.state == 'blind' and error_diff_mag < self.max_divergence:
            print('A target is in view, switching to optical tracking')
            self.state = 'optical'
        elif self.state == 'optical' and error_diff_mag > self.max_divergence:
            print('Solutions diverged, switching to blind tracking')
            self.state = 'blind'

        return blind_error if self.state == 'blind' else optical_error
