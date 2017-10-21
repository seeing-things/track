import os
import fcntl
import select
import datetime
import multiprocessing
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
            ctrl.value = int(self.ctlval_exposure)
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

        self.proc_exit_r, self.proc_exit_w = os.pipe()

        self.frames_out, self.frames_in = multiprocessing.Pipe(duplex=False)

        self.proc = multiprocessing.Process(target=self.monitor, name='WebCam Process')
        self.proc.daemon = True # docs: "when a process exists, it attempts to kill all of its daemonic child processes"
        self.proc.start()

    def __del__(self):
        if hasattr(self, 'proc') and self.proc.is_alive():
            # first attempt to use the pipe/select/join method to cleanly end the monitor process
            os.write(self.proc_exit_w, '!')
            self.proc.join(1.0)
            # failing that, kill it the brutal way
            if self.proc.is_alive():
                print('WebCam: monitor process didn\'t end; terminating it forcefully')
                self.proc.terminate()
        if hasattr(self, 'proc_exit_r'):
            os.close(self.proc_exit_r)
        if hasattr(self, 'proc_exit_w'):
            os.close(self.proc_exit_w)
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
    # TODO: add timeout parameter: if webcam process dies for some reason, this will block
    # indefinitely and therefore make the main thread (in other parts of the codebase) hang badly
    def get_fresh_frame(self):
        # block until there's at least one frame in the pipe
        self.frames_out.poll(None)

        frames = []
        while self.frames_out.poll():
            frames += [self.frames_out.recv()]

        # return only the most recently acquired frame
        return frames[-1]

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

    # [WEBCAM PROCESS] webcam monitor loop
    def monitor(self):
        try:
            while True:
                readable, writable, exceptional = select.select((self.camera, self.proc_exit_r), (), (), 0.5)

                # the main thread is asking us to terminate
                if self.proc_exit_r in readable:
                    return

                frames_jpeg = []
                frames_bgr  = []

                # grab as many frames as are available to us right now
                while self.has_frames_available():
                    now = datetime.datetime.utcnow() # TODO
                    frames_jpeg += [self.camera.read_and_queue()]

                # decode the compressed JPEG image data into BGR format for OpenCV's use
                for frame_jpeg in frames_jpeg:
                    frames_bgr += [cv2.imdecode(np.fromstring(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)]

                # send the decoded frames across the pipe to the main process
                for frame_bgr in frames_bgr:
                    self.frames_in.send(frame_bgr)
        except KeyboardInterrupt:
            print('Webcam process caught KeyboardInterrupt')
            return

    # [WEBCAM PROCESS] does the webcam have at least one frame ready for us to read? (non-blocking)
    def has_frames_available(self):
        readable, writable, exceptional = select.select((self.camera,), (), (), 0.0)
        return (self.camera in readable)
