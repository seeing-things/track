import fcntl
import select
import numpy
import v4l2capture    # USE THIS FORK: https://github.com/gebart/python-v4l2capture
from PIL import Image # Use the Pillow fork, not the original/old PIL library


class WebCam(object):

    def __init__(self, dev_path, res_wanted, num_buffers):
        self.dev_path    = dev_path
        self.res_wanted  = res_wanted
        self.num_buffers = num_buffers

        self.camera = v4l2capture.Video_device(self.dev_path)

        self.res_actual = self.camera.set_format(self.res_wanted[0], self.res_wanted[1], yuv420=0)

        self.camera.create_buffers(self.num_buffers)
        self.camera.queue_all_buffers()
        self.camera.start()

    def __del__(self):
        self.camera.stop()
        self.camera.close()

    # get the ACTUAL webcam frame width
    def get_res_x(self):
        return self.res_actual[0]

    # get the ACTUAL webcam frame height
    def get_res_y(self):
        return self.res_actual[1]

    # get the most recent frame from the webcam, waiting if necessary, and throwing away stale frames if any
    # (the frame is a numpy array in RGB format)
    def get_fresh_frame(self):
        self.block_until_frame_ready()

        frames = []
        while self.has_frames_available():
            frames += [self.get_one_frame()]

        return frames[-1]

    # get one frame from the webcam buffer; the frame is not guaranteed to be the most recent frame available!
    # (the frame is a numpy array in RGB format)
    def get_one_frame(self):
        frame_raw = self.camera.read_and_queue()

        frame_img = Image.frombytes('RGB', self.res_actual, frame_raw)

        frame = numpy.array(frame_img)
        return frame

    # block until the webcam has at least one frame ready
    def block_until_frame_ready(self):
        select.select((self.camera,), (), ())

    # query whether the webcam has at least one frame ready for us to read (non-blocking)
    def has_frames_available(self):
        readable, writable, exceptional = select.select((self.camera,), (), (), 0.0)
        return (len(readable) != 0)
