import sys
import os
import errno
import fcntl
import select
import time
import datetime
import numpy as np
import cv2
import v4l2
import v4l2capture # USE THIS FORK: https://github.com/gebart/python-v4l2capture


class WebCam(object):

    def __init__(self, dev_path, num_buffers, ctlval_exposure):
        self.dev_path        = dev_path
        self.num_buffers     = num_buffers
        self.ctlval_exposure = ctlval_exposure

        self.dev = open(self.dev_path, 'r')

        # set the JPEG compression quality to the maximum level possible
        try:
            jpegcomp = v4l2.v4l2_jpegcompression()
            fcntl.ioctl(self.dev, v4l2.VIDIOC_G_JPEGCOMP, jpegcomp)
            jpegcomp.quality = 100
            fcntl.ioctl(self.dev, v4l2.VIDIOC_S_JPEGCOMP, jpegcomp)
        except (IOError, OSError):
            print('WebCam: failed to set control: JPEG compression quality')

        # disable automatic gain control
        try:
            ctrl = v4l2.v4l2_control()
            ctrl.id    = v4l2.V4L2_CID_AUTOGAIN
            ctrl.value = 0
            fcntl.ioctl(self.dev, v4l2.VIDIOC_S_CTRL, ctrl)
        except (IOError, OSError):
            print('WebCam: failed to set control: automatic gain')

        # set exposure to the desired level
        try:
            ctrl = v4l2.v4l2_control()
            ctrl.id    = v4l2.V4L2_CID_EXPOSURE
            ctrl.value = self.ctlval_exposure
            fcntl.ioctl(self.dev, v4l2.VIDIOC_S_CTRL, ctrl)
        except (IOError, OSError):
            print('WebCam: failed to set control: exposure')

        # ensure that the device supports 'JFIF JPEG' format video capture
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.dev, v4l2.VIDIOC_G_FMT, fmt)
        assert (fmt.fmt.pix.pixelformat == v4l2.V4L2_PIX_FMT_JPEG)

        # sanity-check the allegedly supported camera width and height
        assert (fmt.fmt.win.w.left > 0 and fmt.fmt.win.w.left <= 10240)
        assert (fmt.fmt.win.w.top  > 0 and fmt.fmt.win.w.top  <= 10240)
        self.res_wanted = (fmt.fmt.win.w.left, fmt.fmt.win.w.top)

        self.camera = v4l2capture.Video_device(self.dev_path)

        self.res_actual = self.camera.set_format(self.res_wanted[0], self.res_wanted[1], yuv420=0, fourcc='JPEG')

        self.camera.create_buffers(self.num_buffers)
        self.camera.queue_all_buffers()
        self.camera.start()

        self.dump_init()

    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
            self.camera.close()
        if hasattr(self, 'dev'):
            self.dev.close()

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
        return cv2.imdecode(np.fromstring(frames[-1], dtype=np.uint8), cv2.IMREAD_COLOR)

    # get one frame from the webcam buffer; the frame is not guaranteed to be the most recent frame available!
    # (the frame is a JPEG byte string)
    def get_one_frame(self):
        jpeg = self.camera.read_and_queue()
        self.dump_one(jpeg)
        return jpeg

    # block until the webcam has at least one frame ready
    def block_until_frame_ready(self):
        select.select((self.camera,), (), ())

    # query whether the webcam has at least one frame ready for us to read (non-blocking)
    def has_frames_available(self):
        readable, writable, exceptional = select.select((self.camera,), (), (), 0.0)
        return (len(readable) != 0)

    def is_ctrl_supported(self, id):
        query = v4l2.v4l2_queryctrl()
        query.id = id
        try:
            fcntl.ioctl(self.dev, v4l2.VIDIOC_QUERYCTRL, query)
        except (IOError, OSError):
            return False
        if ((query.flags & v4l2.V4L2_CTRL_FLAG_DISABLED) != 0):
            return False
        return True

    def dump_init(self):
        self.dump_idx = 0

        # find and create a not-yet-existent 'dump####' directory
        num = 0
        while True:
            self.dump_dir = 'dump{:04d}'.format(num)
            try:
                os.makedirs(self.dump_dir)
            except (IOError, OSError) as e:
                if e.errno == errno.EEXIST:
                    num += 1
                else:
                    raise
            else:
                break

    def dump_one(self, jpeg):
        file_name = 'frame_{:04d}_{:%Y%m%d_%H%M%S_%f}.jpg'.format(self.dump_idx, datetime.utcnow())
        self.dump_idx += 1

        file_path = os.path.join(self.dump_dir, file_name)

        if sys.version_info >= (3,3): # Python 3.3+ added 'x' exclusive mode
            mode = 'wbx'
        else:                         # otherwise: manually prevent overwrite
            mode = 'wb'
            assert not os.path.exists(file_path)

        with open(file_path, mode) as f:
            f.write(jpeg)
