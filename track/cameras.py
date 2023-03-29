"""Cameras for use in telescope tracking control loop.

A set of classes that inherit from the abstract base class Camera, providing a common API for
interacting with cameras. The main application for a camera relevant to this package is as a
sensor to be paired with computer vision algorithms that can estimate the position error of a
target.
"""

from __future__ import annotations
from abc import abstractmethod
from contextlib import AbstractContextManager
import logging
from typing import Tuple
from math import inf
import enum
import os
import time
import fcntl
import select
import mmap
import errno
import ctypes
import numpy as np
import v4l2
from configargparse import Namespace
import cv2
from astropy.coordinates import Angle
import astropy.units as u
import asi
from asi import ASICheck, ASIError
from track.config import ArgParser


logger = logging.getLogger(__name__)


class Camera(AbstractContextManager):
    """Abstract base class for cameras"""

    @property
    @abstractmethod
    def pixel_scale(self) -> float:
        """Field of view of a single pixel.

        This is a property of the camera, the optical system to which it is attached. The value
        returned should be correct for the size of the physical photosites in the camera sensor,
        which may be different from the effective size of pixels returned by get_frame() if binning
        is enabled.

        Returns:
            Scale of a pixel in degrees per pixel.
        """

    @property
    @abstractmethod
    def binning(self) -> int:
        """Binning configuration.

        Many cameras support binning of multiple physical photosites in a square grid into a single
        value as if they were a single photosite. The binning number is the number of photosites
        on each side of such a grid, so a binning of 2 means photosites are grouped into 2x2 grids
        of four photosites each. A binning value of 1 means that no binning is applied and all
        photosites are preserved.

        Returns:
            Binning number.
        """

    @property
    @abstractmethod
    def field_of_view(self) -> Tuple[float, float]:
        """Field of view of the camera.

        This is a function of the camera physical sensor size and the focal length of the optical
        system to which it is attached.

        Returns:
            A tuple (height, width) giving the field of view in degrees.
        """

    @property
    @abstractmethod
    def frame_shape(self) -> Tuple[int, int]:
        """Dimensions of the frame in pixels.

        This should be identical to the .shape property of the arrays returned by get_frame(). It
        should therefore account for any processing that occurs between the camera sensor and
        get_frame() that alters the resolution such as binning or region of interest (ROI).

        Returns:
            Resolution as a tuple of (height, width) in pixels.
        """

    @property
    @abstractmethod
    def video_mode(self) -> bool:
        """Indicate if camera is in video mode.

        Returns:
            True if the camera is in video mode. False otherwise.
        """

    @video_mode.setter
    @abstractmethod
    def video_mode(self, enabled: bool) -> None:
        """Enable or disable video mode if camera supports this.

        Raises:
            ValueError if the camera does not support the requested mode.
        """

    @abstractmethod
    def get_frame(self, timeout: float = inf) -> np.ndarray:
        """Get a frame from the camera.

        This method should return the latest frame from the camera. It should be implemented such
        that the same frame is not returned more than once. If no new frame is available since the
        previous call it should block until either a new frame arrives or the timeout expires. If
        multiple frames have arrived since the last call only the most recent should be returned
        and the others should be dropped. It is the responsibility of the caller to call frequently
        enough to avoid dropping frames if frame loss is a problem.

        Args:
            timeout: How long to wait for a frame in seconds.

        Returns:
            Latest frame from the camera or None if the timeout expires before a frame arrives. The
            shape of the array should match the frame_shape property.
        """

    @staticmethod
    @abstractmethod
    def add_program_arguments(parser: ArgParser) -> None:
        """Adds program arguments specific to this camera.

        This method should add program arguments required by this camera to the passed-in ArgParser
        instance.
        """

    def pixel_in_frame(self, position_x: float, position_y: float) -> bool:
        """Check if a pixel location is inside the frame.

        Args:
            position_x: Position along the horizontal axis, where 0 is the center of the left-most
                pixel and positive is right.
            position_y: Position along the vertical axis, where 0 is the center of the top-most
                pixel and positive is down.

        Returns:
            True if the pixel location is within the frame. False otherwise.
        """
        frame_height, frame_width = self.frame_shape
        # Each pixel spans from -0.5 to +0.5 about any integer position
        return (-0.5 <= position_x <= frame_width - 0.5) and (
            -0.5 <= position_y <= frame_height - 0.5
        )

    def pixel_to_angle(self, position_x: float, position_y: float) -> Angle:
        """Convert a pixel location to a complex angle from frame center.

        Args:
            position_x: Position along the horizontal axis, where 0 is the center of the left-most
                pixel and positive is right.
            position_y: Position along the vertical axis, where 0 is the center of the top-most
                pixel and positive is down.

        Returns:
            An Angle representing the position in a coordinate system where the center of the frame
            is the origin with the positive real axis pointing to the right and the positive
            imaginary axis pointing up. The magnitude of the angle represents the separation
            angle between the center of the frame and this position in the sky.
        """
        frame_height, frame_width = self.frame_shape

        # pixel location of the center of the camera frame
        center_px_x = frame_width / 2.0 - 0.5
        center_px_y = frame_height / 2.0 - 0.5

        # angular separation from the frame center along X and Y axes separately
        target_x = Angle((position_x - center_px_x) * self.pixel_scale * self.binning * u.deg)
        target_y = Angle((center_px_y - position_y) * self.pixel_scale * self.binning * u.deg)

        return target_x + 1j * target_y


class CameraTimeout(Exception):
    """Raised when a timeout expires"""


class ASICamera(Camera):
    """ZWO ASI Cameras"""

    class BitDepth(enum.IntEnum):
        """Bit depth of pixels."""

        RAW8 = asi.ASI_IMG_RAW8
        RAW16 = asi.ASI_IMG_RAW16

        def bytes_per_pixel(self) -> int:
            """Number of bytes per pixel in raw array of frame data retrieved from ASI driver"""
            return 1 if self == self.RAW8 else 2

    @staticmethod
    def add_program_arguments(parser: ArgParser) -> None:
        """Adds program arguments for ZWO ASI camera configuration.

        Args:
            parser: The instance of ArgParser to which this function will add arguments.
        """
        parser.add_argument(
            '--zwo-exposure-time',
            help='ZWO camera exposure time used during tracking in seconds',
            default=0.03,
            type=float,
        )
        parser.add_argument(
            '--zwo-gain', help='ZWO camera gain used during tracking', default=10, type=int
        )
        parser.add_argument('--zwo-binning', help='ZWO camera binning', default=4, type=int)
        parser.add_argument(
            '--zwo-name',
            help='ZWO camera name (use to select between multiple connected cameras)',
            type=str,
        )

    @staticmethod
    def from_program_args(args: Namespace) -> 'ASICamera':
        """Factory to make a WebCam instance from program arguments

        Args:
            args: Set of program arguments.

        Returns:
            An instance of ASICamera initialized with the appropriate configuration.
        """
        camera = ASICamera(
            pixel_scale=args.camera_pixel_scale / 3600.0,
            binning=args.zwo_binning,
            name=args.zwo_name,
        )
        camera.exposure = args.zwo_exposure_time
        camera.gain = args.zwo_gain
        return camera

    def __init__(
        self,
        pixel_scale: float,
        binning: int = 1,
        video_mode: bool = False,
        name: str = None,
    ):
        """Initialize and configure ZWO ASI camera.

        Args:
            pixel_scale: Scale of a pixel in degrees per pixel before binning.
            binning: Camera binning.
            video_mode: False for one-shot mode, True for video mode.
            name: Connect only to a camera where the Name member of the info struct matches this
                string. If None the name is not checked and the first camera is used. Note that
                multiple cameras could have the same Name; when this is the case, this constructor
                will connect to the first camera having a Name that matches.

        Raises:
            ASIError for any camera related problems.
        """
        num_connected = asi.ASIGetNumOfConnectedCameras()
        if num_connected == 0:
            raise ASIError('No cameras connected')

        # find the right camera
        self.info = None
        for idx in range(num_connected):
            # pylint does not seem to handle SWIG bindings perfectly
            # pylint: disable=no-value-for-parameter
            info = ASICheck(asi.ASIGetCameraProperty(idx))
            if name is None or name == info.Name:
                self.info = info
                break

        if self.info is None:
            raise ASIError(f'Could not find a camera with name "{name}"')

        self._pixel_scale = pixel_scale
        self._binning = binning
        width = self.info.MaxWidth // binning
        height = self.info.MaxHeight // binning
        self._frame_shape = (height, width)
        ASICheck(asi.ASIOpenCamera(self.info.CameraID))
        ASICheck(asi.ASIInitCamera(self.info.CameraID))
        ASICheck(asi.ASISetControlValue(self.info.CameraID, asi.ASI_MONO_BIN, 1, asi.ASI_FALSE))
        ASICheck(
            asi.ASISetControlValue(self.info.CameraID, asi.ASI_BANDWIDTHOVERLOAD, 94, asi.ASI_FALSE)
        )
        self.video_mode = video_mode

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Close camera if it's open."""
        if hasattr(self, 'info') and self.info is not None:
            ASICheck(asi.ASICloseCamera(self.info.CameraID))
            logger.info(f'Closed camera {self.info.Name}')
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            logger.info(f'Handling {type(exc_value).__name__}')
            return True  # prevent exception propagation
        return False

    def _set_ctrl(self, ctrl, value: int):
        # auto mode always disabled since we generally don't trust it
        ASICheck(asi.ASISetControlValue(self.info.CameraID, ctrl, value, asi.ASI_FALSE))

    def _get_ctrl(self, ctrl):
        return ASICheck(asi.ASIGetControlValue(self.info.CameraID, ctrl))

    @property
    def pixel_scale(self) -> float:
        """Scale of a pixel in degrees per pixel"""
        return self._pixel_scale

    @property
    def binning(self) -> int:
        """Binning configuration"""
        return self._binning

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """Shape of array returned by get_frame()"""
        return self._frame_shape

    @property
    def field_of_view(self) -> Tuple[float, float]:
        """Field of view of the camera (height, width) in degrees."""
        return (self._pixel_scale * self.info.MaxHeight, self._pixel_scale * self.info.MaxWidth)

    @property
    def video_mode(self) -> bool:
        """True if video mode is enabled"""
        return self._video_mode

    @video_mode.setter
    def video_mode(self, enabled: bool) -> None:
        """Enable or disable video mode"""
        self._bit_depth = self.BitDepth.RAW8 if enabled else self.BitDepth.RAW16
        height, width = self._frame_shape
        self._frame_size_bytes = width * height * self._bit_depth.bytes_per_pixel()
        ASICheck(
            asi.ASISetROIFormat(self.info.CameraID, width, height, self._binning, self._bit_depth)
        )
        if enabled:
            ASICheck(
                asi.ASISetControlValue(
                    self.info.CameraID, asi.ASI_HIGH_SPEED_MODE, 1, asi.ASI_FALSE
                )
            )
            ASICheck(asi.ASIStartVideoCapture(self.info.CameraID))
        else:
            ASICheck(
                asi.ASISetControlValue(
                    self.info.CameraID, asi.ASI_HIGH_SPEED_MODE, 0, asi.ASI_FALSE
                )
            )
            ASICheck(asi.ASIStopVideoCapture(self.info.CameraID))
        self._video_mode = enabled

    @property
    def gain(self) -> int:
        """Camera gain"""
        return self._get_ctrl(asi.ASI_GAIN)[0]

    @gain.setter
    def gain(self, gain: int) -> None:
        """Set camera gain"""
        self._set_ctrl(asi.ASI_GAIN, gain)

    @property
    def exposure(self) -> float:
        """Exposure time in seconds"""
        return self._get_ctrl(asi.ASI_EXPOSURE)[0] / 1e6

    @exposure.setter
    def exposure(self, exposure: float) -> None:
        """Set exposure time in seconds"""
        self._set_ctrl(asi.ASI_EXPOSURE, int(exposure * 1e6))

    def _reshape_frame_data(self, frame: np.ndarray) -> np.ndarray:
        """Reshape raw byte array from ASI driver to a 2D array image"""
        if self._bit_depth == self.BitDepth.RAW16:
            frame = frame.view(dtype=np.uint16)
        return np.reshape(frame, self._frame_shape)

    def get_dropped_frames(self) -> int:
        """Get number of dropped frames so far"""
        return ASICheck(asi.ASIGetDroppedFrames(self.info.CameraID))

    def get_frame(self, timeout: float = inf) -> np.ndarray:
        """Get a frame from the camera. See base class docstring for details."""
        if self.video_mode:
            timeout_ms = int(timeout * 1000) if timeout < inf else -1
            status_code, frame = asi.ASIGetVideoData(
                self.info.CameraID, self._frame_size_bytes, timeout_ms
            )
            if status_code == asi.ASI_ERROR_TIMEOUT:
                return None
            elif status_code != asi.ASI_SUCCESS:
                raise RuntimeError(f'Got status code {status_code} from ASIGetVideoData')

            # If we got one frame with no timeout see if there are any more frames waiting so that
            # the freshest frame is returned. Stale frames are dropped.
            while True:
                status_code, frame_tmp = asi.ASIGetVideoData(
                    self.info.CameraID, self._frame_size_bytes, 0
                )
                if status_code == asi.ASI_SUCCESS:
                    frame = frame_tmp
                elif status_code == asi.ASI_ERROR_TIMEOUT:
                    break
                else:
                    raise RuntimeError(f'Got status code {status_code} from ASIGetVideoData')

            frame = self._reshape_frame_data(frame)
        else:
            frame = self.take_exposure(timeout)

        if self.info.IsColorCam == asi.ASI_TRUE and self._binning == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)

        return frame

    def take_exposure(self, timeout: float = inf) -> np.ndarray:
        """Take an exposure in non-video mode.

        This method will poll the camera driver in a loop until the camera indicates that the
        exposure has completed. To avoid pegging the CPU in this loop, a sleep statement is used
        with a period of 10 ms. This approach should be reasonable for single exposures. Use other
        methods when video is desired.

        Args:
            timeout: How long to wait for exposure to complete in seconds. The default value inf
                can be used to disable timeout.

        Returns:
            A numpy array containing a raw camera frame. No debayering is performed.

        Raises:
            RuntimeError if the exposure failed.
            CameraTimeout if the timeout expires before exposure completes.
        """
        ASICheck(asi.ASIStartExposure(self.info.CameraID, asi.ASI_FALSE))
        start_time = time.perf_counter()
        while True:
            time.sleep(0.01)
            status = ASICheck(asi.ASIGetExpStatus(self.info.CameraID))
            if status == asi.ASI_EXP_SUCCESS:
                break
            if status == asi.ASI_EXP_FAILED:
                raise RuntimeError('Exposure failed')
            if time.perf_counter() - start_time > timeout:
                raise CameraTimeout('Timeout waiting for exposure completion')
        frame = ASICheck(asi.ASIGetDataAfterExp(self.info.CameraID, self._frame_size_bytes))
        return self._reshape_frame_data(frame)


class WebCam(Camera):
    """Webcams or other cameras that can be accessed using the 'Video4Linux' (V4L) drivers."""

    @staticmethod
    def add_program_arguments(parser: ArgParser) -> None:
        """Adds program arguments for Webcam camera configuration.

        Args:
            parser: The instance of ArgParser to which this function will add arguments.
        """
        parser.add_argument('--webcam-dev', help='webcam device node path', default='/dev/video0')
        parser.add_argument(
            '--webcam-exposure',
            help='webcam exposure time (unspecified units)',
            default=3200,
            type=int,
        )
        parser.add_argument(
            '--webcam-frame-dump-dir',
            help='directory to save webcam frames as jpeg files on disk',
        )

    @staticmethod
    def from_program_args(args: Namespace) -> 'WebCam':
        """Factory to make a WebCam instance from program arguments"""
        return WebCam(
            dev_path=args.webcam_dev,
            ctrl_exposure=args.webcam_exposure,
            # program arg is in arcseconds
            pixel_scale=args.camera_pixel_scale / 3600.0,
            frame_dump_dir=args.webcam_frame_dump_dir,
        )

    def __init__(
        self,
        dev_path: str,
        ctrl_exposure: int,
        pixel_scale: float,
        frame_dump_dir: str = None,
    ):
        self.dev_path = dev_path
        self._pixel_scale = pixel_scale
        self.dump_frames_to_files = frame_dump_dir is not None
        self.dev_fd = -1
        self.bufmaps = []
        self.started = False

        self.dev_fd = os.open(self.dev_path, os.O_RDWR | os.O_NONBLOCK)

        # disable autogain before setting exposure
        self._set_autogain(False)
        self._set_exposure(ctrl_exposure)
        self._set_jpeg_quality(100)

        # frame_shape attribute is required by abstract base class
        frame_shape_wanted = self._verify_capabilities()
        self._frame_shape = self._set_format(frame_shape_wanted, v4l2.V4L2_PIX_FMT_JPEG)

        self._setup_buffers(buf_count=15)
        self._queue_all_buffers()

        if self.dump_frames_to_files:
            self._dump_init(frame_dump_dir)

        self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Clean up resources."""
        self.stop()
        for bufmap in self.bufmaps:
            bufmap.close()
        if self.dev_fd != -1:
            os.close(self.dev_fd)
        logger.info('Closed webcam.')
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            logger.info(f'Handling {type(exc_value).__name__}')
            return True  # prevent exception propagation
        return False

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frame_shape

    @property
    def field_of_view(self) -> Tuple[float, float]:
        # pylint: disable=consider-using-generator
        return tuple([self._pixel_scale * side for side in self._frame_shape])

    @property
    def pixel_scale(self) -> float:
        return self._pixel_scale

    @property
    def binning(self) -> int:
        return 1

    @property
    def video_mode(self) -> bool:
        """Always returns True because this camera can only be in video mode"""
        return True

    @video_mode.setter
    def video_mode(self, enabled: bool) -> None:
        """Video mode is always enabled for this camera"""
        if not enabled:
            raise ValueError("Can't disable video mode for Webcam")

    def get_frame(self, timeout: float = inf) -> np.ndarray:
        """Get the most recent frame from the webcam.

        Gets most recent frame, throwing away stale frames if any. The frame will also be saved to
        disk as a JPEG image file if this feature is enabled.

        Args:
            timeout: How long to wait for a fresh frame in seconds. Use math.inf to disable.

        Returns:
            The frame as a numpy array.
        """

        self.block_until_frame_ready(timeout)

        frames = []
        while self.has_frames_available():
            frames += [self.get_one_frame()]
            if self.dump_frames_to_files:
                self._dump_one(frames[-1])

        if len(frames) == 0:
            return None

        # decode the JPEG from the webcam into BGR for OpenCV's use
        color_frame = cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.IMREAD_COLOR)

        # convert to grayscale
        return cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    def get_one_frame(self):
        """Get one frame from the webcam buffer.

        The frame is not guaranteed to be the most recent frame available!

        Returns:
            The frame as a JPEG byte string.
        """
        return self._read_and_queue()

    def block_until_frame_ready(self, timeout: float = inf) -> bool:
        """Block until the webcam has at least one frame ready or timeout expires.

        Args:
            timeout: How long to wait before giving up in seconds or None for no timeout.

        Returns:
            True if a frame is ready or False if timeout expired.
        """
        if timeout < inf:
            dev_fd_ready, _, _ = select.select((self.dev_fd,), (), (), timeout=timeout)
        else:
            dev_fd_ready, _, _ = select.select((self.dev_fd,), (), ())
        return len(dev_fd_ready) > 0

    def has_frames_available(self) -> bool:
        """query whether the webcam has at least one frame ready for us to read (non-blocking)"""
        readable, _, _ = select.select((self.dev_fd,), (), (), 0.0)
        return len(readable) != 0

    def start(self) -> None:
        """tell the camera to start capturing"""
        if not self.started:
            self._v4l2_ioctl(
                v4l2.VIDIOC_STREAMON, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
            )
            self.started = True

    def stop(self) -> None:
        """tell the camera to stop capturing"""
        if self.started:
            self._v4l2_ioctl(
                v4l2.VIDIOC_STREAMOFF, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
            )
            self.started = False

    def _enum_pixel_formats(self):
        return self._enum_common(
            v4l2.VIDIOC_ENUM_FMT,
            lambda idx: v4l2.v4l2_fmtdesc(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index=idx,
            ),
        )

    def _enum_frame_sizes(self, pixel_format):
        # not handling types other than V4L2_FRMSIZE_TYPE_DISCRETE
        return self._enum_common(
            v4l2.VIDIOC_ENUM_FRAMESIZES,
            lambda idx: v4l2.v4l2_frmsizeenum(
                index=idx,
                pixel_format=pixel_format,
            ),
        )

    def _enum_frame_intervals(self, pixel_format, width, height):
        # not handling types other than V4L2_FRMIVAL_TYPE_DISCRETE
        return self._enum_common(
            v4l2.VIDIOC_ENUM_FRAMEINTERVALS,
            lambda idx: v4l2.v4l2_frmivalenum(
                index=idx,
                pixel_format=pixel_format,
                width=width,
                height=height,
            ),
        )

    def _enum_common(self, req, l_getreq):
        idx = 0
        results = []
        while True:
            request = l_getreq(idx)
            try:
                self._v4l2_ioctl(req, request)
            except OSError as e:
                if e.errno == errno.EINVAL:
                    break
                raise
            results += [request]
            idx += 1
        return results

    def _verify_capabilities(self) -> Tuple[int, int]:
        fmt = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)

        # ensure that the device supports 'JFIF JPEG' format video capture
        assert fmt.fmt.pix.pixelformat == v4l2.V4L2_PIX_FMT_JPEG

        # sanity-check the allegedly supported camera width and height
        assert fmt.fmt.win.w.left > 0 and fmt.fmt.win.w.left <= 10240
        assert fmt.fmt.win.w.top > 0 and fmt.fmt.win.w.top <= 10240

        # return supported resolution
        return (fmt.fmt.win.w.top, fmt.fmt.win.w.left)

    def _set_exposure(self, level: int) -> None:
        self._set_ctrl(v4l2.V4L2_CID_EXPOSURE, int(level), 'exposure level')

    def _set_autogain(self, enable: bool) -> None:
        self._set_ctrl(v4l2.V4L2_CID_AUTOGAIN, bool(enable), 'automatic gain')

    def _set_ctrl(self, ctrl_id, value, desc):
        ctrl = v4l2.v4l2_control(id=ctrl_id, value=value)
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_S_CTRL, ctrl, f'failed to set control: {desc}')

    def _set_jpeg_quality(self, quality):
        jpegcomp = v4l2.v4l2_jpegcompression()
        self._v4l2_ioctl_nonfatal(
            v4l2.VIDIOC_G_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality'
        )
        jpegcomp.quality = quality
        self._v4l2_ioctl_nonfatal(
            v4l2.VIDIOC_S_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality'
        )

    def _set_format(self, shape_wanted: Tuple[int, int], fourcc) -> Tuple[int, int]:
        """roughly equivalent to v4l2capture's set_format"""
        assert not self.started

        fmt = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)

        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = shape_wanted[1]
        fmt.fmt.pix.height = shape_wanted[0]
        fmt.fmt.pix.bytesperline = 0
        fmt.fmt.pix.pixelformat = fourcc
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_ANY
        self._v4l2_ioctl(v4l2.VIDIOC_S_FMT, fmt)

        # return actual resolution
        return (fmt.fmt.pix.height, fmt.fmt.pix.width)

    def _setup_buffers(self, buf_count: int) -> None:
        """roughly equivalent to v4l2capture's create_buffers"""
        assert not self.started
        assert len(self.bufmaps) == 0

        reqbuf = v4l2.v4l2_requestbuffers(
            type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, count=buf_count, memory=v4l2.V4L2_MEMORY_MMAP
        )
        self._v4l2_ioctl(v4l2.VIDIOC_REQBUFS, reqbuf)
        assert reqbuf.count > 0

        for idx in range(reqbuf.count):
            buf = v4l2.v4l2_buffer(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, index=idx, memory=v4l2.V4L2_MEMORY_MMAP
            )
            self._v4l2_ioctl(v4l2.VIDIOC_QUERYBUF, buf)
            self.bufmaps += [
                mmap.mmap(self.dev_fd, buf.length, access=mmap.ACCESS_WRITE, offset=buf.m.offset)
            ]

    def _queue_all_buffers(self) -> None:
        """roughly equivalent to v4l2capture's queue_all_buffers"""
        assert not self.started
        assert len(self.bufmaps) != 0

        for idx in range(len(self.bufmaps)):
            buf = v4l2.v4l2_buffer(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, index=idx, memory=v4l2.V4L2_MEMORY_MMAP
            )
            self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

    def _read_and_queue(self):
        """roughly equivalent to v4l2capture's read_and_queue"""
        assert self.started

        buf = v4l2.v4l2_buffer(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=v4l2.V4L2_MEMORY_MMAP)
        self._v4l2_ioctl(v4l2.VIDIOC_DQBUF, buf)

        frame = self.bufmaps[buf.index].read(buf.bytesused)
        self.bufmaps[buf.index].seek(0)

        self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

        return frame

    def _v4l2_ioctl_nonfatal(self, req, arg, err_msg):
        try:
            self._v4l2_ioctl(req, arg)
        except OSError:
            logger.exception(f'WebCam ioctl failed: {err_msg}')

    def _v4l2_ioctl(self, req, arg):
        assert fcntl.ioctl(self.dev_fd, req, arg) == 0

    def _dump_init(self, frame_dump_dir: str) -> None:
        self.dump_idx = 0

        # find and create a not-yet-existent 'webcam_dump_####' directory
        num = 0
        while True:
            self.dump_dir = os.path.join(frame_dump_dir, f'webcam_dump_{num:04d}')
            try:
                os.makedirs(self.dump_dir)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    num += 1
                else:
                    raise
            else:
                break

    def _dump_one(self, jpeg):
        file_name = f'frame_{self.dump_idx:06d}.jpg'
        self.dump_idx += 1

        file_path = os.path.join(self.dump_dir, file_name)

        # prevent overwrite
        assert not os.path.exists(file_path)

        with open(file_path, 'wb') as f:
            f.write(jpeg)


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments for all cameras.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    camera_group = parser.add_argument_group(
        title='General Camera Options',
        description='Options that apply to cameras of any type',
    )
    camera_group.add_argument(
        '--camera-type',
        help='type of camera',
        default='zwo',
        choices=['zwo', 'webcam'],
    )
    camera_group.add_argument(
        '--camera-pixel-scale',
        help='camera pixel scale in arcseconds per pixel',
        required=True,
        type=float,
    )
    webcam_group = parser.add_argument_group(
        title='Webcam Options',
        description='Options that apply when camera-type is set to "webcam"',
    )
    WebCam.add_program_arguments(webcam_group)
    zwo_group = parser.add_argument_group(
        title='ZWO ASI Camera Options',
        description='Options that apply when camera-type is set to "zwo"',
    )
    ASICamera.add_program_arguments(zwo_group)


def make_camera_from_args(args: Namespace, video_mode: bool = False) -> Camera:
    """Construct the appropriate camera based on the program arguments provided.

    Args:
        args: Set of program arguments.
        video_mode: Puts camera in video mode when True.
    """
    if args.camera_type == 'webcam':
        camera = WebCam.from_program_args(args)
    elif args.camera_type == 'zwo':
        camera = ASICamera.from_program_args(args)
    else:
        raise ValueError(f'Invalid camera type {args.camera_type}')
    camera.video_mode = video_mode
    return camera
