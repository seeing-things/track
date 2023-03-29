"""Classes for controlling focuser hardware.

A set of classes that inherit from the abstract base class Focuser, providing a common API for
interacting with focusers. The main application in this package is autofocus, which is faster,
typically more reliable, and certainly less frustrating than focusing the camera manually.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing import Optional
import serial
from configargparse import Namespace
from track.config import ArgParser


class Focuser(ABC):
    """Abstract base class for focusers"""

    @property
    @abstractmethod
    def min_position(self) -> int:
        """Minimum allowed focuser position."""

    @property
    @abstractmethod
    def max_position(self) -> int:
        """Maximum allowed focuser position."""

    @property
    @abstractmethod
    def current_position(self) -> int:
        """Get current position of the focuser."""

    @current_position.setter
    @abstractmethod
    def current_position(self, position: int) -> None:
        """Set current position.

        This sets the register containing the current position value. It is not a request to move
        to a new position. This is typically used at startup to set the zero position such that it
        corresponds to a meaningful physical focuser position. For example, it may be desirable to
        enforce that position 0 corresponds to the draw tube being fully retracted.
        """

    @property
    @abstractmethod
    def target_position(self) -> int:
        """Get target position of the focuser."""

    @target_position.setter
    @abstractmethod
    def target_position(self, position: int) -> None:
        """Set target position.

        This sets the register containing the desired (target) position value. The motor will not
        move to this position until `move_to_target_position()` is called.
        """

    @property
    @abstractmethod
    def motor_speed(self) -> int:
        """Get motor speed."""

    @motor_speed.setter
    @abstractmethod
    def motor_speed(self, speed: int) -> None:
        """Set motor speed."""

    @property
    @abstractmethod
    def in_motion(self) -> bool:
        """Query if motor is in motion."""

    @abstractmethod
    def move_to_target_position(self, blocking: bool = False) -> None:
        """Move to the target position.

        Args:
            blocking: If True, this call will block until the focuser has arrived at the new
                position. Otherwise it returns immediately.
        """

    @abstractmethod
    def stop_motion(self) -> None:
        """Stop motion immediately."""


class MoonliteFocuser(Focuser):
    """Implements the MoonLite High Res Stepper Motor serial command set."""

    class ResponseException(Exception):
        """Raised on bad command responses from the motor.

        Attributes:
            response (bytes): The byte array which was received but was deemed invalid. This can be
                inspected to see what the (possibly partial) bad response looks like. Do not expect
                this buffer to necessarily match the expected response length; it could be shorter
                or longer.
        """

        def __init__(self, response, *args, **kwargs):
            self.response = response
            super().__init__(*args, **kwargs)

    class ReadTimeoutException(Exception):
        """Raised when read from the motor times out."""

    @staticmethod
    def add_program_arguments(parser: ArgParser) -> None:
        """Add Moonlite-specific program arguments"""
        parser.add_argument(
            '--focuser-dev', help='Moonlite focuser serial device node path', default='/dev/ttyUSB0'
        )

    @staticmethod
    def from_program_args(args: Namespace) -> MoonliteFocuser:
        """Factory to make a MoonliteFocuser instance from program arguments"""
        return MoonliteFocuser(
            device=args.focuser_dev,
            min_position=args.focuser_min,
            max_position=args.focuser_max,
        )

    def __init__(
        self, device: str, min_position: int, max_position: int, read_timeout: float = 1.0
    ):
        """Constructs a MoonliteFocuser object.

        Args:
            device: The path to the serial device. For example, '/dev/ttyUSB0'.
            min_position: Minimum allowed position. Must greater than or equal to 0.
            max_position: Maximum allowed position. Must be less than or equal to 0xffff (65535).
            read_timeout: Timeout in seconds for reads on the serial device.

        Raises:
            ValueError if `min_position` or `max_position` are outside allowed range.
        """
        if not 0 <= min_position <= 0xFFFF:
            raise ValueError('min_position is beyond allowed range')
        if not 0 <= max_position <= 0xFFFF:
            raise ValueError('max_position is beyond allowed range')
        if max_position < min_position:
            raise ValueError('max_position is less than min position')
        self._min_position = min_position
        self._max_position = max_position
        self._serial = serial.Serial(device, baudrate=9600, timeout=read_timeout)

    def _send_command(self, command: bytearray, response_len: int) -> Optional[bytearray]:
        """Sends a command to the focuser and reads back the response.

        Args:
            command: A byte array containing the ASCII command to send, excluding the start and
                termination characters.
            response_len: An integer giving the expected length of the response to this command,
                not counting the terminating '#' character.

        Returns:
            A byte array containing the response from the focuser, excluding the termination
                character.

        Raises:
            ReadTimeoutException: When a timeout occurs during the attempt to read from the serial
                device.
            ResponseException: When the response length does not match the value of the
                response_len argument.
        """

        # Eliminate any stale data sitting in the read buffer which could be left over from prior
        # command responses.
        self._serial.read(self._serial.in_waiting)

        self._serial.write(b':' + command + b'#')

        if response_len == 0:
            return None

        response = self._serial.read_until(terminator=b'#')
        if response[-1:] != b'#':
            raise MoonliteFocuser.ReadTimeoutException()

        # strip off the '#' terminator
        response = response[:-1]
        if b'#' in response:
            raise MoonliteFocuser.ResponseException(
                response, 'Unexpected terminator found in response'
            )

        if len(response) != response_len:
            raise MoonliteFocuser.ResponseException(
                response,
                f'Expected response length {response_len} but got {len(response)} instead.',
            )

        return response

    @property
    def min_position(self) -> int:
        return self._min_position

    @property
    def max_position(self) -> int:
        return self._max_position

    @property
    def current_position(self) -> int:
        """Get current position of the stepper motor."""
        return int(self._send_command(b'GP', 4), 16)

    @current_position.setter
    def current_position(self, position: int) -> None:
        """Set current position.

        This sets the register containing the current position value. It is not a request to move
        to a new position. This is typically used at startup to set the zero position such that it
        corresponds to a meaningful physical focuser position. For example, it may be desirable to
        enforce that position 0 corresponds to the draw tube being fully retracted.
        """
        if not self._min_position <= position <= self._max_position:
            raise ValueError('position is beyond allowed range')
        self._send_command(f'SP{position:04X}'.encode(), 0)

    @property
    def target_position(self) -> int:
        """Get target position of the stepper motor."""
        return int(self._send_command(b'GN', 4), 16)

    @target_position.setter
    def target_position(self, position: int) -> None:
        """Set target position.

        This sets the register containing the desired (target) position value. The motor will not
        move to this position until `move_to_new_position()` is called.
        """
        if not self._min_position <= position <= self._max_position:
            raise ValueError('position is beyond allowed range')
        self._send_command(f'SN{position:04X}'.encode(), 0)

    @property
    def temperature(self) -> int:
        """Get current temperature."""
        return int(self._send_command(b'GT', 4), 16)

    @property
    def motor_speed(self) -> int:
        """Get motor speed. Valid speeds are 2, 4, 8, 16, and 32."""
        return int(self._send_command(b'GD', 2), 16)

    @motor_speed.setter
    def motor_speed(self, speed: int) -> None:
        """Set motor speed. Valid speeds are 2, 4, 8, 16, and 32. Lower values are faster."""
        if speed not in [2, 4, 8, 16, 32]:
            raise ValueError('invalid speed')
        self._send_command(f'SD{speed:02X}'.encode(), 0)

    @property
    def half_step_mode(self) -> bool:
        """True if half step mode, false otherwise."""
        return self._send_command(b'GH', 2) == b'FF'

    @half_step_mode.setter
    def half_step_mode(self, half_step_mode: bool) -> None:
        """Set half step mode."""
        if half_step_mode:
            self._send_command(b'SH', 0)
        else:
            self._send_command(b'SF', 0)

    @property
    def in_motion(self) -> bool:
        """Query if motor is in motion."""
        return self._send_command(b'GI', 2) == b'01'

    @property
    def led_backlight_val(self) -> int:
        """Get red LED backlight value."""
        return int(self._send_command(b'GB', 2), 16)

    @property
    def firmware_version(self) -> int:
        """Get firmware version code."""
        return int(self._send_command(b'GV', 2), 16)

    def move_to_target_position(self, blocking: bool = False) -> None:
        """Move to the target position.

        Args:
            blocking: If True, this call will block until the focuser has arrived at the new
                position. Otherwise it returns immediately.
        """
        self._send_command(b'FG', 0)
        if blocking:
            while self.in_motion:
                time.sleep(0.1)

    def stop_motion(self) -> None:
        """Stop motor immediately."""
        self._send_command(b'FQ', 0)


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments for all focusers.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    focuser_group = parser.add_argument_group(
        title='General Focuser Options',
        description='Options that apply to all focusers',
    )
    focuser_group.add_argument(
        '--focuser-type',
        help='type of focuser',
        default='moonlite',
        choices=['moonlite'],
    )
    focuser_group.add_argument(
        '--focuser-min', help='focuser minimum allowed position', required=True, type=int
    )
    focuser_group.add_argument(
        '--focuser-max', help='focuser maximum allowed position', required=True, type=int
    )
    moonlite_group = parser.add_argument_group(
        title='Moonlite Focuser Options',
        description='Options that apply when focuser-type is set to "moonlite"',
    )
    MoonliteFocuser.add_program_arguments(moonlite_group)


def make_focuser_from_args(args: Namespace) -> Focuser:
    """Construct the appropriate focuser based on the program arguments provided.

    Args:
        args: Set of program arguments.
    """
    if args.focuser_type == 'moonlite':
        return MoonliteFocuser.from_program_args(args)
    raise ValueError(f'Invalid focuser-type {args.focuser_type}')
