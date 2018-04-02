import os
import fcntl
import select
import mmap
import errno
import ctypes
import numpy as np
import cv2
import v4l2


class WebCam(object):

    def __init__(self, dev_path, bufs_wanted, ctrl_exposure):
        self.dev_path    = dev_path
        self.bufs_wanted = bufs_wanted
        self.dev_fd      = -1
        self.bufmaps     = []
        self.started     = False

        # detect OpenCV version to handle API differences between 2 and 3
        self.opencv_ver = int(cv2.__version__.split('.')[0])
        assert (self.opencv_ver == 2 or self.opencv_ver == 3)

        self.dev_fd = os.open(self.dev_path, os.O_RDWR | os.O_NONBLOCK);

        self.res_wanted = self._verify_capabilities()

        self._set_exposure(ctrl_exposure)
        self._set_autogain(False)
        self._set_jpeg_quality(100)

        self.res_actual = self._set_format(self.res_wanted, v4l2.V4L2_PIX_FMT_JPEG)

        self._setup_buffers(self.bufs_wanted)
        self._queue_all_buffers()

        self.start()

    def __del__(self):
        self.stop()
        for bufmap in self.bufmaps:
            bufmap.close()
        if self.dev_fd != -1:
            os.close(self.dev_fd)

    # get the ACTUAL webcam frame width
    def get_res_x(self):
        return self.res_actual[0]

    # get the ACTUAL webcam frame height
    def get_res_y(self):
        return self.res_actual[1]

    # get the most recent frame from the webcam, waiting if necessary, and throwing away stale frames if any
    # (the frame is a numpy array in BGR format)
    def get_fresh_frame(self):
        self.block_until_frame_ready()

        frames = []
        while self.has_frames_available():
            frames += [self.get_one_frame()]

        # decode the JPEG from the webcam into BGR for OpenCV's use
        if self.opencv_ver == 2:
            return cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        if self.opencv_ver == 3:
            return cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.IMREAD_COLOR)

    # get one frame from the webcam buffer; the frame is not guaranteed to be the most recent frame available!
    # (the frame is a JPEG byte string)
    def get_one_frame(self):
        return self._read_and_queue()

    # block until the webcam has at least one frame ready
    def block_until_frame_ready(self):
        select.select((self.dev_fd,), (), ())

    # query whether the webcam has at least one frame ready for us to read (non-blocking)
    def has_frames_available(self):
        readable, writable, exceptional = select.select((self.dev_fd,), (), (), 0.0)
        return (len(readable) != 0)

    # tell the camera to start capturing
    def start(self):
        if not self.started:
            self._v4l2_ioctl(v4l2.VIDIOC_STREAMON, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)))
            self.started = True

    # tell the camera to stop capturing
    def stop(self):
        if self.started:
            self._v4l2_ioctl(v4l2.VIDIOC_STREAMOFF, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)))
            self.started = False


    def _verify_capabilities(self):
        fmt      = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)
        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE pixel formats
        #       and not just the current one
        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE resolutions
        #       and not just the current one

        # ensure that the device supports 'JFIF JPEG' format video capture
        assert (fmt.fmt.pix.pixelformat == v4l2.V4L2_PIX_FMT_JPEG)

        # sanity-check the allegedly supported camera width and height
        assert (fmt.fmt.win.w.left > 0 and fmt.fmt.win.w.left <= 10240)
        assert (fmt.fmt.win.w.top  > 0 and fmt.fmt.win.w.top  <= 10240)

        # return supported resolution
        return [fmt.fmt.win.w.left, fmt.fmt.win.w.top]

    def _set_exposure(self, level):  self._set_ctrl(v4l2.V4L2_CID_EXPOSURE, int (level),  'exposure level')
    def _set_autogain(self, enable): self._set_ctrl(v4l2.V4L2_CID_AUTOGAIN, bool(enable), 'automatic gain')

    def _set_ctrl(self, id, value, desc):
        ctrl       = v4l2.v4l2_control()
        ctrl.id    = id
        ctrl.value = value
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_S_CTRL, ctrl, 'failed to set control: {}'.format(desc))

    def _set_jpeg_quality(self, quality):
        jpegcomp         = v4l2.v4l2_jpegcompression()
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_G_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality')
        jpegcomp.quality = quality
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_S_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality')

    # roughly equivalent to v4l2capture's set_format
    def _set_format(self, res_wanted, fourcc):
        assert (not self.started)

        fmt      = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)

        fmt.type                 = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width        = res_wanted[0]
        fmt.fmt.pix.height       = res_wanted[1]
        fmt.fmt.pix.bytesperline = 0
        fmt.fmt.pix.pixelformat  = fourcc
        fmt.fmt.pix.field        = v4l2.V4L2_FIELD_ANY
        self._v4l2_ioctl(v4l2.VIDIOC_S_FMT, fmt)

        # return actual resolution
        return [fmt.fmt.pix.width, fmt.fmt.pix.height]

    # roughly equivalent to v4l2capture's create_buffers
    def _setup_buffers(self, buf_count):
        assert (not self.started)
        assert (len(self.bufmaps) == 0)

        reqbuf        = v4l2.v4l2_requestbuffers()
        reqbuf.count  = buf_count
        reqbuf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        reqbuf.memory = v4l2.V4L2_MEMORY_MMAP
        self._v4l2_ioctl(v4l2.VIDIOC_REQBUFS, reqbuf)
        assert (reqbuf.count > 0)

        for idx in range(reqbuf.count):
            buf        = v4l2.v4l2_buffer()
            buf.index  = idx
            buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            self._v4l2_ioctl(v4l2.VIDIOC_QUERYBUF, buf)

            self.bufmaps += [mmap.mmap(self.dev_fd, buf.length, access=mmap.ACCESS_WRITE, offset=buf.m.offset)]

    # roughly equivalent to v4l2capture's queue_all_buffers
    def _queue_all_buffers(self):
        assert (not self.started)
        assert (len(self.bufmaps) != 0)

        for idx in range(len(self.bufmaps)):
            buf        = v4l2.v4l2_buffer()
            buf.index  = idx
            buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

    # roughly equivalent to v4l2capture's read_and_queue
    def _read_and_queue(self):
        assert (self.started)

        buf        = v4l2.v4l2_buffer()
        buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
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
        assert (fcntl.ioctl(self.dev_fd, req, arg) == 0)
