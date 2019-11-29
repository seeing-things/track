from abc import ABC, abstractmethod
from typing import Optional, Tuple
from math import inf
import os
import fcntl
import select
import mmap
import errno
import ctypes
import numpy as np
import cv2
import v4l2


class Camera(ABC):
    """Abstract base class for cameras"""


    @property
    @abstractmethod
    def pixel_scale(self) -> float:
        """Field of view of a single pixel.

        This is a property of the camera, the optical system to which it is attached. The value
        returned should be correct for the size of the pixels in the frame returned by the
        get_frame() method, which may be different from the size of the camera sensor photosites if
        binning is enabled.

        Returns:
            Scale of a pixel in degrees per pixel.
        """


    @property
    @abstractmethod
    def frame_shape(self) -> Tuple[int, int]:
        """Dimensions of the frame in pixels.

        This should be identical to the .shape property of the arrays returned by get_frame(). It
        should therefore account for any processing that occurs between the camera sensor and
        get_frame() that alters the resolution such as binning or region of interest (ROI).

        Returns:
            Resolution as a tuple of (width, height) in pixels.
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


class WebCam(Camera):

    def __init__(self,
            dev_path: str,
            bufs_wanted: int,
            ctrl_exposure: int,
            pixel_scale: float,
            frame_dump_dir: str = None,
        ):

        self.dev_path = dev_path
        self.bufs_wanted = bufs_wanted
        self.pixel_scale = pixel_scale
        self.dump_frames_to_files = frame_dump_dir is not None
        self.dev_fd = -1
        self.bufmaps = []
        self.started = False

        # detect OpenCV version to handle API differences between 2 and 3
        self.opencv_ver = int(cv2.__version__.split('.')[0])
        assert (self.opencv_ver == 2 or self.opencv_ver == 3)

        self.dev_fd = os.open(self.dev_path, os.O_RDWR | os.O_NONBLOCK)

        # disable autogain before setting exposure
        self._set_autogain(False)
        self._set_exposure(ctrl_exposure)
        self._set_jpeg_quality(100)

        # frame_shape attribute is required by abstract base class
        frame_shape_wanted = self._verify_capabilities()
        self.frame_shape = self._set_format(frame_shape_wanted, v4l2.V4L2_PIX_FMT_JPEG)

        self._setup_buffers(self.bufs_wanted)
        self._queue_all_buffers()

        if self.dump_frames_to_files:
            self._dump_init(frame_dump_dir)

        self.start()


    def __del__(self):
        self.stop()
        for bufmap in self.bufmaps:
            bufmap.close()
        if self.dev_fd != -1:
            os.close(self.dev_fd)


    def get_frame(self, timeout: Optional[float] = None) -> np.ndarray:
        """Get the most recent frame from the webcam.

        Gets most recent frame, throwing away stale frames if any. The frame will also be saved to
        disk as a JPEG image file if this feature is enabled.

        Args:
            timeout: How long to wait for a fresh frame in seconds or None for no timeout.

        Returns:
            The frame as a numpy array in BGR format.
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
        if self.opencv_ver == 2:
            return cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        if self.opencv_ver == 3:
            return cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.IMREAD_COLOR)

    # get one frame from the webcam buffer; the frame is not guaranteed to be the most recent frame
    # available! (the frame is a JPEG byte string)
    def get_one_frame(self):
        return self._read_and_queue()

    def block_until_frame_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the webcam has at least one frame ready or timeout expires.

        Args:
            timeout: How long to wait before giving up in seconds or None for no timeout.

        Returns:
            True if a frame is ready or False if timeout expired.
        """
        if timeout is not None:
            dev_fd_ready, _, _ = select.select((self.dev_fd,), (), (), timeout=timeout)
            return False if len(dev_fd_ready) == 0 else True
        else:
            select.select((self.dev_fd,), (), ())
            return True

    # query whether the webcam has at least one frame ready for us to read (non-blocking)
    def has_frames_available(self):
        readable, _, _ = select.select((self.dev_fd,), (), (), 0.0)
        return (len(readable) != 0)

    # tell the camera to start capturing
    def start(self):
        if not self.started:
            self._v4l2_ioctl(
                v4l2.VIDIOC_STREAMON,
                ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
            )
            self.started = True

    # tell the camera to stop capturing
    def stop(self):
        if self.started:
            self._v4l2_ioctl(
                v4l2.VIDIOC_STREAMOFF,
                ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
            )
            self.started = False


    def _enum_pixel_formats(self):
        return self._enum_common(v4l2.VIDIOC_ENUM_FMT,
            lambda idx: v4l2.v4l2_fmtdesc(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index=idx,
            )
        )

    def _enum_frame_sizes(self, pixel_format):
        # TODO: handle types other than V4L2_FRMSIZE_TYPE_DISCRETE sanely
        return self._enum_common(v4l2.VIDIOC_ENUM_FRAMESIZES,
            lambda idx: v4l2.v4l2_frmsizeenum(
                index=idx,
                pixel_format=pixel_format,
            )
        )

    def _enum_frame_intervals(self, pixel_format, width, height):
        # TODO: handle types other than V4L2_FRMIVAL_TYPE_DISCRETE sanely
        return self._enum_common(v4l2.VIDIOC_ENUM_FRAMEINTERVALS,
            lambda idx: v4l2.v4l2_frmivalenum(
                index=idx,
                pixel_format=pixel_format,
                width=width,
                height=height,
            )
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
                else:
                    raise
            results += [request]
            idx += 1
        return results


    def _verify_capabilities(self):
        fmt = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)
        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE pixel
        #       formats and not just the current one
        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE
        #       resolutions and not just the current one

        # ensure that the device supports 'JFIF JPEG' format video capture
        assert fmt.fmt.pix.pixelformat == v4l2.V4L2_PIX_FMT_JPEG

        # sanity-check the allegedly supported camera width and height
        assert fmt.fmt.win.w.left > 0 and fmt.fmt.win.w.left <= 10240
        assert fmt.fmt.win.w.top  > 0 and fmt.fmt.win.w.top  <= 10240

        # return supported resolution
        return [fmt.fmt.win.w.left, fmt.fmt.win.w.top]

    def _set_exposure(self, level):
        self._set_ctrl(v4l2.V4L2_CID_EXPOSURE, int (level),  'exposure level')


    def _set_autogain(self, enable):
        self._set_ctrl(v4l2.V4L2_CID_AUTOGAIN, bool(enable), 'automatic gain')


    def _set_ctrl(self, id, value, desc):
        ctrl = v4l2.v4l2_control(id=id, value=value)
        self._v4l2_ioctl_nonfatal(
            v4l2.VIDIOC_S_CTRL,
            ctrl,
            'failed to set control: {}'.format(desc)
        )


    def _set_jpeg_quality(self, quality):
        jpegcomp = v4l2.v4l2_jpegcompression()
        self._v4l2_ioctl_nonfatal(
            v4l2.VIDIOC_G_JPEGCOMP,
            jpegcomp,
            'failed to set JPEG compression quality'
        )
        jpegcomp.quality = quality
        self._v4l2_ioctl_nonfatal(
            v4l2.VIDIOC_S_JPEGCOMP,
            jpegcomp,
            'failed to set JPEG compression quality'
        )


    # roughly equivalent to v4l2capture's set_format
    def _set_format(self, res_wanted, fourcc):
        assert not self.started

        fmt = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)

        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = res_wanted[0]
        fmt.fmt.pix.height = res_wanted[1]
        fmt.fmt.pix.bytesperline = 0
        fmt.fmt.pix.pixelformat = fourcc
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_ANY
        self._v4l2_ioctl(v4l2.VIDIOC_S_FMT, fmt)

        # return actual resolution
        return [fmt.fmt.pix.width, fmt.fmt.pix.height]

    # roughly equivalent to v4l2capture's create_buffers
    def _setup_buffers(self, buf_count):
        assert not self.started
        assert len(self.bufmaps) == 0

        reqbuf = v4l2.v4l2_requestbuffers(
            type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
            count=buf_count,
            memory=v4l2.V4L2_MEMORY_MMAP
        )
        self._v4l2_ioctl(v4l2.VIDIOC_REQBUFS, reqbuf)
        assert (reqbuf.count > 0)

        for idx in range(reqbuf.count):
            buf = v4l2.v4l2_buffer(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index=idx,
                memory=v4l2.V4L2_MEMORY_MMAP
            )
            self._v4l2_ioctl(v4l2.VIDIOC_QUERYBUF, buf)
            self.bufmaps += [mmap.mmap(
                self.dev_fd,
                buf.length,
                access=mmap.ACCESS_WRITE,
                offset=buf.m.offset
            )]

    # roughly equivalent to v4l2capture's queue_all_buffers
    def _queue_all_buffers(self):
        assert not self.started
        assert len(self.bufmaps) != 0

        for idx in range(len(self.bufmaps)):
            buf = v4l2.v4l2_buffer(
                type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index=idx,
                memory=v4l2.V4L2_MEMORY_MMAP
            )
            self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

    # roughly equivalent to v4l2capture's read_and_queue
    def _read_and_queue(self):
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
            print('WebCam: {}'.format(err_msg))

    def _v4l2_ioctl(self, req, arg):
        assert fcntl.ioctl(self.dev_fd, req, arg) == 0

    def _dump_init(self, frame_dump_dir):
        self.dump_idx = 0

        # find and create a not-yet-existent 'webcam_dump_####' directory
        num = 0
        while True:
            self.dump_dir = frame_dump_dir + '/webcam_dump_{:04d}'.format(num)
            try:
                os.makedirs(self.dump_dir)
            except (IOError, OSError) as e:
                if e.errno == errno.EEXIST:
                    num += 1
                else:
                    raise
            else:
                break

    def _dump_one(self, jpeg):
        file_name = 'frame_{:06d}.jpg'.format(self.dump_idx)
        self.dump_idx += 1

        file_path = os.path.join(self.dump_dir, file_name)

        # prevent overwrite
        assert not os.path.exists(file_path)

        with open(file_path, 'wb') as f:
            f.write(jpeg)
