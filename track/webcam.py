import os
import fcntl
import select
import mmap
import errno
import ctypes
import numpy as np
import cv2
import v4l2
import sys


class WebCam(object):

    def __init__(self, dev_path, bufs_wanted, ctrl_exposure, dump_frames_to_files=False):
        self.dev_path    = dev_path
        self.bufs_wanted = bufs_wanted
        self.dump_frames_to_files = dump_frames_to_files
        self.dev_fd      = -1
        self.bufmaps     = []
        self.started     = False

        # detect OpenCV version to handle API differences between 2 and 3
        self.opencv_ver = int(cv2.__version__.split('.')[0])
        assert (self.opencv_ver == 2 or self.opencv_ver == 3)

        self.dev_fd = os.open(self.dev_path, os.O_RDWR | os.O_NONBLOCK);

        self.res_wanted = self._verify_capabilities()

        # disable autogain before setting exposure
        self._set_autogain(False)
        self._set_exposure(ctrl_exposure)
        self._set_jpeg_quality(100)

        self.res_actual = self._set_format(self.res_wanted, v4l2.V4L2_PIX_FMT_JPEG)

        self._setup_buffers(self.bufs_wanted)
        self._queue_all_buffers()

        if self.dump_frames_to_files:
            self._dump_init()

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

    # Get the most recent frame from the webcam, waiting if blocking is True, and throwing away
    # stale frames if any (the frame is a numpy array in BGR format). If enabled, all queued frames
    # will also be saved as JPEG image files on disk.
    def get_fresh_frame(self, blocking=True):

        if blocking:
            self.block_until_frame_ready()

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


    def _enum_pixel_formats(self):
        return self._enum_common(v4l2.VIDIOC_ENUM_FMT,
            lambda idx: v4l2.v4l2_fmtdesc(
                type  = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index = idx,
            )
        )

    def _enum_frame_sizes(self, pixel_format):
        # TODO: handle types other than V4L2_FRMSIZE_TYPE_DISCRETE sanely
        return self._enum_common(v4l2.VIDIOC_ENUM_FRAMESIZES,
            lambda idx: v4l2.v4l2_frmsizeenum(
                index        = idx,
                pixel_format = pixel_format,
            )
        )

    def _enum_frame_intervals(self, pixel_format, width, height):
        # TODO: handle types other than V4L2_FRMIVAL_TYPE_DISCRETE sanely
        return self._enum_common(v4l2.VIDIOC_ENUM_FRAMEINTERVALS,
            lambda idx: v4l2.v4l2_frmivalenum(
                index        = idx,
                pixel_format = pixel_format,
                width        = width,
                height       = height,
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
        ctrl = v4l2.v4l2_control(id=id, value=value)
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_S_CTRL, ctrl, 'failed to set control: {}'.format(desc))

    def _set_jpeg_quality(self, quality):
        jpegcomp = v4l2.v4l2_jpegcompression()
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_G_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality')
        jpegcomp.quality = quality
        self._v4l2_ioctl_nonfatal(v4l2.VIDIOC_S_JPEGCOMP, jpegcomp, 'failed to set JPEG compression quality')

    # roughly equivalent to v4l2capture's set_format
    def _set_format(self, res_wanted, fourcc):
        assert (not self.started)

        fmt = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
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

        reqbuf = v4l2.v4l2_requestbuffers(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, count=buf_count, memory=v4l2.V4L2_MEMORY_MMAP)
        self._v4l2_ioctl(v4l2.VIDIOC_REQBUFS, reqbuf)
        assert (reqbuf.count > 0)

        for idx in range(reqbuf.count):
            buf = v4l2.v4l2_buffer(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, index=idx, memory=v4l2.V4L2_MEMORY_MMAP)
            self._v4l2_ioctl(v4l2.VIDIOC_QUERYBUF, buf)
            self.bufmaps += [mmap.mmap(self.dev_fd, buf.length, access=mmap.ACCESS_WRITE, offset=buf.m.offset)]

    # roughly equivalent to v4l2capture's queue_all_buffers
    def _queue_all_buffers(self):
        assert (not self.started)
        assert (len(self.bufmaps) != 0)

        for idx in range(len(self.bufmaps)):
            buf = v4l2.v4l2_buffer(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE, index=idx, memory=v4l2.V4L2_MEMORY_MMAP)
            self._v4l2_ioctl(v4l2.VIDIOC_QBUF, buf)

    # roughly equivalent to v4l2capture's read_and_queue
    def _read_and_queue(self):
        assert (self.started)

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
        assert (fcntl.ioctl(self.dev_fd, req, arg) == 0)

    def _dump_init(self):
        self.dump_idx = 0

        # find and create a not-yet-existent 'webcam_dump_####' directory
        num = 0
        while True:
            self.dump_dir = 'webcam_dump_{:04d}'.format(num)
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
        file_name = 'frame_{:04d}.jpg'.format(self.dump_idx)
        self.dump_idx += 1

        file_path = os.path.join(self.dump_dir, file_name)

        # prevent overwrite
        assert not os.path.exists(file_path)

        with open(file_path, 'wb') as f:
            f.write(jpeg)