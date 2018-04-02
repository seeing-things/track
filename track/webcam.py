import os
import fcntl
import select
import mmap
import ctypes
import numpy as np
import cv2
import v4l2


class WebCam(object):

    def __init__(self, dev_path, num_buffers, ctrlval_exposure):
        self.dev_path    = dev_path
        self.num_buffers = num_buffers

        # detect OpenCV version to handle API differences between 2 and 3
        self.opencv_ver = int(cv2.__version__.split('.')[0])
        assert (self.opencv_ver == 2 or self.opencv_ver == 3)

        self.dev_fd = os.open(self.dev_path, os.O_RDWR | os.O_NONBLOCK);

        self._set_exposure(ctrlval_exposure)
        self._set_autogain(False)
        self._set_jpeg_quality(100)

        fmt = self._get_capture_format()

        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE pixfmts and not just the current one
        # ensure that the device supports 'JFIF JPEG' format video capture
        assert (fmt.pix.pixelformat == v4l2.V4L2_PIX_FMT_JPEG)

        # TODO: check whether VIDIOC_G_FMT is even the right IOCTL for determining POSSIBLE resolutions and not just the current one
        # sanity-check the allegedly supported camera width and height
        assert (fmt.win.w.left > 0 and fmt.win.w.left <= 10240)
        assert (fmt.win.w.top  > 0 and fmt.win.w.top  <= 10240)
        self.res_wanted = (fmt.win.w.left, fmt.win.w.top)

        self.camera = V4L2WebCam(self.dev_fd)

        fmt = self.camera.set_format(self.res_wanted[0], self.res_wanted[1], fourcc=v4l2.V4L2_PIX_FMT_JPEG)
        self.res_actual_x = fmt.fmt.pix.width
        self.res_actual_y = fmt.fmt.pix.height

        self.camera.create_buffers(self.num_buffers)
        self.camera.queue_all_buffers()
        self.camera.start()

    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
            self.camera.close()
        if hasattr(self, 'dev_fd'):
            os.close(self.dev_fd)

    # get the ACTUAL webcam frame width
    def get_res_x(self):
        return self.res_actual_x

    # get the ACTUAL webcam frame height
    def get_res_y(self):
        return self.res_actual_y

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
#        return self.camera.read_and_queue()
        x = self.camera.read_and_queue()
        return x

    # block until the webcam has at least one frame ready
    def block_until_frame_ready(self):
        select.select((self.dev_fd,), (), ())

    # query whether the webcam has at least one frame ready for us to read (non-blocking)
    def has_frames_available(self):
        readable, writable, exceptional = select.select((self.dev_fd,), (), (), 0.0)
        return (len(readable) != 0)


    def _set_exposure(self, level):
        self._set_ctrl(v4l2.V4L2_CID_EXPOSURE, int(level), 'exposure level')

    def _set_autogain(self, enable):
        self._set_ctrl(v4l2.V4L2_CID_AUTOGAIN, int(enable), 'automatic gain')

    def _set_ctrl(self, id, value, desc):
        try:
            data = v4l2.v4l2_control()
            data.id    = id
            data.value = value
            self._v4l2_ioctl(v4l2.VIDIOC_S_CTRL, data)
        except (IOError, OSError):
            print('WebCam: failed to set control: {}'.format(desc))

    def _set_jpeg_quality(self, quality):
        try:
            # get current JPEG settings first; then change quality; then set
            data = v4l2.v4l2_jpegcompression()
            self._v4l2_ioctl(v4l2.VIDIOC_G_JPEGCOMP, data)
            data.quality = quality
            self._v4l2_ioctl(v4l2.VIDIOC_S_JPEGCOMP, data)
        except (IOError, OSError):
            print('WebCam: failed to set JPEG compression quality')

    def _get_capture_format(self):
        data = v4l2.v4l2_format()
        data.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, data)
        return data.fmt

    def _v4l2_ioctl(self, req, arg):
        fcntl.ioctl(self.dev_fd, req, arg)


# TODO: eliminate this and integrate it into WebCam, if feasible & non-ugly
class V4L2WebCam(object):

    def __init__(self, dev_fd):
        self.dev_fd     = dev_fd
        self.buffers    = []

    def __del__(self):
        self._unmap()

    def set_format(self, size_x, size_y, fourcc):
        # size_x, size_y, fourcc='JPEG'

        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE

        self._v4l2_ioctl(v4l2.VIDIOC_G_FMT, fmt)

        # TODO: double-check the sanity of some of the values below...
        fmt.type                 = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width        = size_x
        fmt.fmt.pix.height       = size_y
        fmt.fmt.pix.bytesperline = 0
        fmt.fmt.pix.pixelformat  = fourcc
        fmt.fmt.pix.field        = v4l2.V4L2_FIELD_ANY

        self._v4l2_ioctl(v4l2.VIDIOC_S_FMT, fmt)

        return fmt

    def create_buffers(self, buf_count):
        reqbuf = v4l2.v4l2_requestbuffers()
        reqbuf.count  = buf_count
        reqbuf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        reqbuf.memory = v4l2.V4L2_MEMORY_MMAP
        self._v4l2_ioctl(v4l2.VIDIOC_REQBUFS, reqbuf)
        assert (reqbuf.count > 0)

        for idx in range(reqbuf.count):
            buf = v4l2.v4l2_buffer()
            buf.index  = idx
            buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            self._v4l2_ioctl(v4l2.VIDIOC_QUERYBUF, buf)

            self.buffers += [mmap.mmap(self.dev_fd, buf.length, access=mmap.ACCESS_WRITE, offset=buf.m.offset)]

    def queue_all_buffers(self):
        for idx in range(len(self.buffers)):
            buf = v4l2.v4l2_buffer()
            buf.index  = idx
            buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

    def start(self):
        self._v4l2_ioctl(v4l2.VIDIOC_STREAMON, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)))

    def stop(self):
        self._v4l2_ioctl(v4l2.VIDIOC_STREAMOFF, ctypes.c_int(int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)))

    def read_and_queue(self):
        buf = v4l2.v4l2_buffer()
        buf.type   = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        self._v4l2_ioctl(v4l2.VIDIOC_DQBUF, buf)

        result = self.buffers[buf.index].read(buf.bytesused)
        self.buffers[buf.index].seek(0)

        self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

        return result

    def close(self):
        self._unmap()


    def _unmap(self):
        for mapping in self.buffers:
            mapping.close()
        self.buffers = []

    def _v4l2_ioctl(self, req, arg):
        fcntl.ioctl(self.dev_fd, req, arg)
