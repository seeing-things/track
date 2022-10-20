from typing import Optional
import serial


# Todo: Write an abstract base class
class MoonliteFocuser:
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

    def __init__(self, device, read_timeout=1):
        """Constructs a MoonliteFocuser object.

        Args:
            device (str): The path to the serial device.
                For example, '/dev/ttyUSB0'.
            read_timeout (float): Timeout in seconds for reads on the serial device.
        """
        self.serial = serial.Serial(device, baudrate=9600, timeout=read_timeout)


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
        self.serial.read(self.serial.in_waiting)

        self.serial.write(b':' + command + b'#')

        if response_len == 0:
            return None

        response = self.serial.read_until(terminator=b'#')
        if response[-1:] != b'#':
            raise MoonliteFocuser.ReadTimeoutException()

        # strip off the '#' terminator
        response = response[:-1]
        if b'#' in response:
            raise MoonliteFocuser.ResponseException(response, 'Unexpected terminator found in response')

        if len(response) != response_len:
            raise MoonliteFocuser.ResponseException(
                response,
                f'Expected response length {response_len} but got {len(response)} instead.')

        return response

    def get_position(self) -> int:
        """Get current position of the stepper motor."""
        return int(self._send_command(b'GP', 4), 16)

    def get_new_position(self) -> int:
        """Get new (target) position of the stepper motor."""
        return int(self._send_command(b'GN', 4), 16)

    def get_temperature(self) -> int:
        """Get current temperature."""
        return int(self._send_command(b'GT', 4), 16)

    def get_motor_speed(self) -> int:
        """Get motor speed. Valid speeds are 2, 4, 8, 16, and 32."""
        return int(self._send_command(b'GD', 2), 16)

    def in_half_step_mode(self) -> bool:
        """True if half step mode, false otherwise."""
        return self._send_command(b'GH', 2) == b'FF'

    def in_motion(self) -> bool:
        """Query if motor is in motion."""
        return self._send_command(b'GI', 2) == b'01'

    def get_led_backlight_val(self) -> int:
        """Get red LED backlight value."""
        return int(self._send_command(b'GB', 2), 16)

    def get_firmware_version(self) -> int:
        """Get firmware version code."""
        return int(self._send_command(b'GV', 2), 16)

    def set_current_position(self, position: int) -> None:
        """Set current position.

        This sets the register containing the current position value. It is not a request to move
        to a new position. This is typically used at startup to set the zero position such that it
        corresponds to a meaningful physical focuser position. For example, it may be desirable to
        enforce that position 0 corresponds to the draw tube being fully retracted.
        """
        if not (0 <= position <= 0xffff):
            raise ValueError('position is beyond allowed range')
        self._send_command(f'SP{position:04X}'.encode(), 0)

    def set_new_position(self, position: int) -> None:
        """Set new position.

        This sets the register containing the desired (new) position value. The motor will not move
        to this position until `move_to_new_position()` is called.
        """
        if not (0 <= position <= 0xffff):
            raise ValueError('position is beyond allowed range')
        self._send_command(f'SN{position:04X}'.encode(), 0)

    def set_full_step(self) -> None:
        """Enable full-step mode."""
        self._send_command(b'SF', 0)

    def set_half_step(self) -> None:
        """Enable half-step mode."""
        self._send_command(b'SH', 0)

    def set_motor_speed(self, speed: int) -> None:
        """Set motor speed. Valid speeds are 2, 4, 8, 16, and 32."""
        if speed not in [2, 4, 8, 16, 32]:
            raise ValueError('invalid speed')
        self._send_command(f'SD{speed:02X}'.encode(), 0)

    def move_to_new_position(self) -> None:
        """Move to the new position."""
        self._send_command(b'FG', 0)

    def stop_motion(self) -> None:
        """Stop motor immediately."""
        self._send_command(b'FQ', 0)
