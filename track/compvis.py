"""computer vision algorithms for identifying targets in a camera frame"""

from typing import List, Optional, Tuple
import numpy as np
import cv2


def find_features(frame: np.ndarray) -> List[cv2.KeyPoint]:
    """Find bright features in a camera frame.

    Args:
        frame: A grayscale camera frame with dtype uint8.

    Returns:
        A list of keypoints.
    """

    if frame.dtype != np.uint8:
        # No fundamental reason; just hasn't been implemented
        raise ValueError('Only uint8 frames are supported')

    hist, _ = np.histogram(frame.ravel(), 256, [0, 256])

    # Threshold is placed at the first index after the peak in the histogram where the slope is
    # non-negative. The idea is to keep the threshold as low as possible to allow detection of dim
    # targets while still rejecting as much background noise as possible.
    hist_max_index = np.argmax(hist)
    if hist_max_index == 255:
        return []
    hist_diff = np.diff(hist)
    threshold = np.argmax(hist_diff[hist_max_index:] >= 0) + hist_max_index

    _, thresh = cv2.threshold(
        frame,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # outer contours only
    _, contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    keypoints = []
    for contour in contours:
        moms = cv2.moments(contour)

        # reject contours with less than 2 pixels
        if contour.shape[0] < 2:
            continue

        # find the centroid
        if moms['m00'] > 0.0:
            center = np.array([moms['m10'] / moms['m00'], moms['m01'] / moms['m00']])
        else:
            # if the contour is so small that the internal area is 0 any random pixel in the
            # contour should be close enough
            center = contour[0][0]

        # find the maximum distance between the center and the contour
        radius = np.max(np.linalg.norm(center - contour, axis=2))

        keypoints.append(cv2.KeyPoint(center[0], center[1], 2.0*radius))

    return keypoints


class PreviewWindow:
    """Generates an annotated camera frame for display in an OpenCV window"""

    def __init__(
            self,
            frame_width: int,
            frame_height: int,
            target_position_desired: Optional[Tuple[float, float]] = None,
            set_target_position_desired_on_click: bool = False
        ):
        """Constructs an instance of PreviewWindow.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            target_position_desired: Desired position of the target within the camera frame as a
                tuple where the first value is the X position in pixels from the left and the
                second value is the Y position in pixels from the top. If None the center of the
                frame is used by default.
            set_target_position_desired_on_click: Allow the desired position of the target within
                the camera frame to be updated by mouse click on the preview window.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', frame_width, frame_height)

        if target_position_desired is None:
            self.target_position_desired = self.frame_center_px
        else:
            self.target_position_desired = target_position_desired
        if set_target_position_desired_on_click:
            cv2.setMouseCallback('frame', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, userdata):
        """OpenCV mouse event callback to set desired target position in the frame

        See OpenCV High-level GUI documentation for description of arguments
        E.g., for version 3.2.0: https://docs.opencv.org/3.2.0/d7/dfc/group__highgui.html
        """

        if flags & cv2.EVENT_FLAG_LBUTTON:  # left mouse button is down
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                self.target_position_desired = (x, y)

    def show_annotated_frame(
            self,
            frame: np.ndarray,
            keypoints: Optional[List[cv2.KeyPoint]] = None,
            target_keypoint: Optional[cv2.KeyPoint] = None
        ) -> None:
        """Displays camera frame in a window with features circled and crosshairs.

        Args:
            frame: Camera frame.
            keypoints: List of all keypoints.
            target_keypoint: The keypoint identified as the target.
        """

        frame_annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # add grey crosshairs
        cv2.line(
            frame_annotated,
            (int(self.frame_center_px[0]), 0),
            (int(self.frame_center_px[0]), self.frame_height - 1),
            (100, 100, 100),
            1
        )
        cv2.line(
            frame_annotated,
            (0, int(self.frame_center_px[1])),
            (self.frame_width - 1, int(self.frame_center_px[1])),
            (100, 100, 100),
            1
        )

        # add cross at desired position of the target
        cv2.drawMarker(
            frame_annotated,
            (int(self.target_position_desired[0]), int(self.target_position_desired[1])),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
        )

        if keypoints is not None:
            # lower bound on keypoint size so the annotation is visible (okay to modify size because
            # we only really care about the center)
            for point in keypoints:
                point.size = max(point.size, 10.0)

            # cicle non-target keypoints in blue
            frame_annotated = cv2.drawKeypoints(
                frame_annotated,
                keypoints,
                np.array([]),
                (255, 0, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        # circle target keypoint in red
        if target_keypoint is not None:
            target_keypoint.size = max(target_keypoint.size, 10.0)
            frame_annotated = cv2.drawKeypoints(
                frame_annotated,
                [target_keypoint],
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        # display the frame in a window
        cv2.imshow('frame', frame_annotated)
        cv2.waitKey(1)
