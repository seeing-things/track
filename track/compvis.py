"""computer vision algorithms for identifying targets in a camera frame"""

from typing import List, Optional
import numpy as np
import cv2


def find_features(frame: np.ndarray) -> List[cv2.KeyPoint]:
    """Find bright features in a camera frame.

    Args:
        frame: A camera frame.

    Returns:
        A list of keypoints.
    """

    # TODO: Make magic numbers parameters or find them adaptively
    # pick a threshold as the max of these two methods:
    # - 99th percentile of histogram
    # - peak of histogram plus a magic number
    hist, _ = np.histogram(frame.ravel(), 256, [0, 256])
    cumsum = np.cumsum(hist)
    threshold_1 = np.argmax(cumsum >= 0.99*cumsum[-1])
    threshold_2 = np.argmax(hist) + 4
    threshold = max(threshold_1, threshold_2)

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
        if moms['m00'] == 0.0:
            continue

        # find the center of mass
        center = np.array([moms['m10'] / moms['m00'], moms['m01'] / moms['m00']])

        # find the maximum distance between the center and the contour
        radius = 0.0
        for p in contour:
            dist = np.linalg.norm(center - p)
            radius = max(radius, dist)

        keypoints.append(cv2.KeyPoint(center[0], center[1], 2.0*radius))

    return keypoints


class PreviewWindow:
    """Generates an annotated camera frame for display in an OpenCV window"""

    def __init__(self, frame_width, frame_height):
        """Constructs an instance of PreviewWindow.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_px = (frame_width / 2.0, frame_height / 2.0)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', frame_width, frame_height)

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

        frame_annotated = frame.copy()

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
