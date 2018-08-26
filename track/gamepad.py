"""Gamepad support.

Defines a single class Gamepad that provide support for game controller interfaces. When run as a
program the integrated x- and y- values are printed to the console.
"""

from __future__ import print_function
import threading
import time
import selectors
import inputs
import numpy as np
from track.telem import TelemSource

class Gamepad(TelemSource):
    """Class for interfacing with gamepads.

    This class implements some useful functionality to allow a human to control
    the telescope either fully manually or to interact with a control loop
    by providing correction feedback.

    Attributes:
        left_x: A floating point value representing the x-axis position of the
            left analog stick with values normalized to [-1.0, +1.0].
        left_y: A floating point value representing the y-axis position of the
            left analog stick with values normalized to [-1.0, +1.0].
        right_x: A floating point value representing the x-axis position of the
            right analog stick with values normalized to [-1.0, +1.0].
        right_y: A floating point value representing the y-axis position of the
            right analog stick with values normalized to [-1.0, +1.0].
        int_x: The integrated value of the x-axes of both analog sticks.
        int_y: The integrated value of the y-axes of both analog sticks.
        left_gain: The gain applied to the left analog stick integrator input.
        right_gain: The gain applied to the right analog stick integrator input.
        int_loop_period: The period of the integrator thread loop in seconds.
        int_limit: The integrators will be limited to this absolute value.
        callbacks: A dict where keys are event codes and values are callback function handles.
        gamepad: An instance of a gamepad object from the inputs package.
        input_thread: A thread reading input from the gamepad.
        integrator_thread: A thread for integrating the analog stick values.
        integrator_mode: A boolean set to True when integrator mode is active.
        running: Threads will stop executing when this is set to False.
        sel: An object of type selectors.BaseSelector used to check if gamepad has data to read.
    """

    MAX_ANALOG_VAL = 2**15
    MIN_LEVEL = 0.01

    def __init__(
            self,
            left_gain=1.0,
            right_gain=0.1,
            int_limit=1.0,
            int_loop_period=0.01
        ):
        """Inits Gamepad object.

        Initializes a Gamepad object and starts two daemon threads to read
        input from the gamepad and to integrate the analog stick inputs. This
        will use the first gamepad found by the inputs package.

        Args:
            left_gain: Gain for the left analog stick integrator input.
            right_gain: Gain for the right analog stick integrator input.
            int_limit: Absolute value of integrators will be limited to this.
            int_loop_period: Period in seconds for the integrator thead loop.
        """
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.int_x = 0.0
        self.int_y = 0.0
        self.left_gain = left_gain
        self.right_gain = right_gain
        self.int_loop_period = int_loop_period
        self.int_limit = int_limit
        self.callbacks = {}
        if len(inputs.devices.gamepads) < 1:
            raise RuntimeError('No gamepads found')
        self.gamepad = inputs.devices.gamepads[0]
        self.input_thread = threading.Thread(target=self.__get_input)
        self.integrator_thread = threading.Thread(target=self.__integrator)
        self.integrator_mode = False
        self.running = True

        # Use a selector on the character device that the inputs package reads from so that we
        # can avoid blocking on calls to gamepad.read() in the input_thread loop. Calls that block
        # indefinitely make it impossible to stop the thread which then makes clean program
        # shutdown difficult or impossible. Daemon threads were used previously but daemon threads
        # are not shut down cleanly.
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.gamepad._character_device, selectors.EVENT_READ)

        self.input_thread.start()
        self.integrator_thread.start()


    def stop(self):
        """Stops the threads from running."""
        self.running = False

    def get_proportional(self):
        """Returns a tuple containing the instantaneous x/y values."""
        x = np.clip(self.left_gain*self.left_x + self.right_gain*self.right_x, -1.0, 1.0)
        y = np.clip(self.left_gain*self.left_y + self.right_gain*self.right_y, -1.0, 1.0)
        return (x, y)

    def get_integrator(self):
        """Returns a tuple containing the integrated x/y values."""
        return (self.int_x, self.int_y)

    def get_value(self):
        """Returns a tuple containing instantaneous or integrated x/y values based on mode."""
        if self.integrator_mode:
            return self.get_integrator()
        else:
            return self.get_proportional()

    def register_callback(self, event_code=None, callback=None):
        """Register a callback function to be called when a particular gamepad event occurs.

        Args:
            event_code: The event code as a string. For example, if set to 'ABS_X', the callback
                function will be called anytime an event with that code id detected.
            callback: Function to be called. The function should take a single argument which will
                be set to event.state for the matching event code. Set to None to remove the
                callback for this code. Only one callback can be registered per event code.
        """
        self.callbacks[event_code] = callback

    def __get_input(self):
        """Thread function for reading input from gamepad

        The inputs package is written such that calls to read() block until an
        event is available to be returned. There is no way to query the device
        to see if any events are ready. Therefore we must execute this loop
        in its own thread so it doesn't block other processing.
        """
        while True:
            if not self.running:
                return

            if self.sel.select(timeout=0.01):
                events = self.gamepad.read()
                for event in events:
                    if event.code == 'ABS_X':
                        left_x = float(event.state) / self.MAX_ANALOG_VAL
                        self.left_x = left_x if abs(left_x) >= self.MIN_LEVEL else 0.0
                    elif event.code == 'ABS_Y':
                        left_y = -float(event.state) / self.MAX_ANALOG_VAL
                        self.left_y = left_y if abs(left_y) >= self.MIN_LEVEL else 0.0
                    elif event.code == 'ABS_RX':
                        right_x = float(event.state) / self.MAX_ANALOG_VAL
                        self.right_x = right_x if abs(right_x) >= self.MIN_LEVEL else 0.0
                    elif event.code == 'ABS_RY':
                        right_y = -float(event.state) / self.MAX_ANALOG_VAL
                        self.right_y = right_y if abs(right_y) >= self.MIN_LEVEL else 0.0
                    elif event.code == 'BTN_NORTH' and event.state == 1:
                        self.int_x = 0.0
                    elif event.code == 'BTN_WEST' and event.state == 1:
                        self.int_y = 0.0
                    elif event.code == 'BTN_TL' and event.state == 1:
                        self.integrator_mode = False
                    elif event.code == 'BTN_TR' and event.state == 1:
                        self.integrator_mode = True

                    # call any callbacks registered for this event.code
                    callback = self.callbacks.get(event.code, None)
                    if callback is not None:
                        callback(event.state)

    def __integrator(self):
        """Thread function for integrating analog stick values.

        Inputs from both analog sticks are multiplied by the respective gains
        and integrated. This is done in a separate thread so that the
        integrators can continue running even when the input thread is blocked
        waiting for new events from the controller.
        """
        while True:
            if not self.running:
                return

            if self.integrator_mode:
                self.int_x += self.left_gain * self.int_loop_period * self.left_x
                self.int_y += self.left_gain * self.int_loop_period * self.left_y
                self.int_x += self.right_gain * self.int_loop_period * self.right_x
                self.int_y += self.right_gain * self.int_loop_period * self.right_y

                self.int_x = np.clip(self.int_x, -self.int_limit, self.int_limit)
                self.int_y = np.clip(self.int_y, -self.int_limit, self.int_limit)

            time.sleep(self.int_loop_period)

    def get_telem_channels(self):
        names = ['left_x', 'left_y', 'right_x', 'right_y', 'int_x', 'int_y']
        chans = {}
        for name in names:
            chans[name] = self.__dict__[name]
        return chans

def main():
    """Prints integrated x/y values to the console every 10 ms."""
    try:
        gamepad = Gamepad()
        while True:
            print('(' + str(gamepad.int_x) + ',' + str(gamepad.int_y) + ')')
            time.sleep(0.01)

    except KeyboardInterrupt:
        print('goodbye')

if __name__ == "__main__":
    main()
