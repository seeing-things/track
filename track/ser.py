"""
The SER file format is popular in astrophotography for storage of RAW images or video. It is
described on this website: http://www.grischa-hahn.homepage.t-online.de/astro/ser/. This class
implements version 3 which is documented here:
http://www.grischa-hahn.homepage.t-online.de/astro/ser/SER%20Doc%20V3b.pdf
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from io import SEEK_END, SEEK_SET
import struct
from typing import Optional, Union
import numpy as np


# Number of ticks from the Visual Basic Date data type to the Unix time epoch. The VB Date type
# is the number of "ticks" since Jan 1, year 0001 in the Gregorian calendar, where each tick is
# 100 ns.
VB_DATE_TICKS_TO_UNIX_EPOCH = 621_355_968_000_000_000
VB_DATE_TICKS_PER_SEC = 10_000_000


class SERColorID(IntEnum):
    """ColorID field of the SER file header"""
    MONO = 0
    BAYER_RGGB = 8
    BAYER_GRBG = 9
    BAYER_GBRG = 10
    BAYER_BGGR = 11
    BAYER_CYYM = 16
    BAYER_YCMY = 17
    BAYER_YMCY = 18
    BAYER_MYYC = 19
    RGB = 100
    BGR = 101


class SEREndianness(IntEnum):
    """Endianness of multi-byte pixel values"""
    BIG_ENDIAN = 0
    LITTLE_ENDIAN = 1

    def dtype_char(self) -> str:
        """Get the endian character ('<' or '>') to use with `numpy.dtype`"""
        return '<' if self == self.LITTLE_ENDIAN else '>'


def make_timestamp_utc(dt: datetime) -> int:
    """Make the UTC timestamp used in the SER header and trailer.

    Args:
        dt: Time to use in this timestamp.

    Returns:
        The integer timestamp value.
    """
    ns_since_epoch = 1e9 * dt.timestamp()
    return (ns_since_epoch // 100) + VB_DATE_TICKS_TO_UNIX_EPOCH


def datetime_from_timestamp_utc(timestamp: int) -> datetime:
    """Make a datetime object from a UTC timestamp used in the SER header or trailer.

    Args:
        timestamp: The raw integer timestamp value from the SER file header or trailer.

    Returns:
        A datetime object representing the timestamp time.
    """
    ns_since_epoch = 100 * (timestamp - VB_DATE_TICKS_TO_UNIX_EPOCH)
    return datetime.fromtimestamp(ns_since_epoch / 1e9, tz=timezone.utc)


class SERHeader:
    """Represents the header portion of an SER file."""

    SIZE_BYTES = 178

    def __init__(self):
        """Create a new `SERHeader` object.

        Only the constant value fields are initialized by the constructor. All other fields should
        be initialized by setting the individual properties of the object after construction or by
        setting the entire raw buffer to a new value.
        """
        self._buffer = bytearray(178)

        # First two fields in buffer are constant values.
        self._buffer[:14] = b'LUCAM-RECORDER'  # Historical artifact of the file format
        self._buffer[14:18] = struct.pack('<i', 0)  # unused field


    @property
    def buffer(self) -> bytearray:
        """Get the raw buffer representing the header exactly as encoded in the SER file."""
        return self._buffer

    @buffer.setter
    def buffer(self, value: Union[bytearray, bytes]) -> None:
        """Set the entire raw buffer to a new value."""
        if len(value) != 178:
            raise ValueError('Buffer is not 178 bytes long')
        (file_id, unused_field) = struct.unpack('<14si', value[:18])
        if file_id != b'LUCAM-RECORDER':
            raise ValueError('Invalid file ID')
        if unused_field != 0:
            raise ValueError('Unexpected value in second unused field of header')
        self._buffer = bytearray(value)

    @property
    def file_id(self) -> bytes:
        """Constant value for all SER file headers. Read only."""
        return bytes(self._buffer[:14])

    @property
    def color_id(self) -> SERColorID:
        """Identifies how color information is encoded."""
        return SERColorID(struct.unpack('<i', self._buffer[18:22])[0])

    @color_id.setter
    def color_id(self, value: SERColorID) -> None:
        self._buffer[18:22] = struct.pack('<i', value)

    @property
    def endianness(self) -> SEREndianness:
        """Indicates whether 16-bit image data is little-endian or big-endian."""
        return SEREndianness(struct.unpack('<i', self._buffer[22:26])[0])

    @endianness.setter
    def endianness(self, value: SEREndianness) -> None:
        self._buffer[22:26] = struct.pack('<i', value)

    @property
    def frame_width(self) -> int:
        """Width of every frame in pixels."""
        return struct.unpack('<i', self._buffer[26:30])[0]

    @frame_width.setter
    def frame_width(self, value: int) -> None:
        self._buffer[26:30] = struct.pack('<i', value)

    @property
    def frame_height(self) -> int:
        """Height of every frame in pixels."""
        return struct.unpack('<i', self._buffer[30:34])[0]

    @frame_height.setter
    def frame_height(self, value: int) -> None:
        self._buffer[30:34] = struct.pack('<i', value)

    @property
    def bit_depth(self) -> int:
        """Number of pits per pixel per color pane (1-16)."""
        return struct.unpack('<i', self._buffer[34:38])[0]

    @bit_depth.setter
    def bit_depth(self, value: int) -> None:
        if not 1 <= value <= 16:
            raise ValueError('Bit depth must be in [1, 16]')
        self._buffer[34:38] = struct.pack('<i', value)

    @property
    def frame_count(self) -> int:
        """Number of image frames in the file."""
        return struct.unpack('<i', self._buffer[38:42])[0]

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        self._buffer[38:42] = struct.pack('<i', value)

    @property
    def observer(self) -> str:
        """Name of observer."""
        return struct.unpack('<40s', self._buffer[42:82])[0].decode('ascii').rstrip('\0')

    @observer.setter
    def observer(self, value: str) -> None:
        """Name of observer. At most 40 ASCII characters."""
        if len(value) > 40:
            raise ValueError('Must be 40 characters or less')
        self._buffer[42:82] = struct.pack('<40s', value.encode('ascii'))

    @property
    def instrument(self) -> str:
        """Name of camera."""
        return struct.unpack('<40s', self._buffer[82:122])[0].decode('ascii').rstrip('\0')

    @instrument.setter
    def instrument(self, value: str) -> None:
        """Name of camera. At most 40 ASCII characters."""
        if len(value) > 40:
            raise ValueError('Must be 40 characters or less')
        self._buffer[82:122] = struct.pack('<40s', value.encode('ascii'))

    @property
    def telescope(self) -> str:
        """Name of telescope."""
        return struct.unpack('<40s', self._buffer[122:162])[0].decode('ascii').rstrip('\0')

    @telescope.setter
    def telescope(self, value: str) -> None:
        """Name of telescope. At most 40 ASCII characters."""
        if len(value) > 40:
            raise ValueError('Must be 40 characters or less')
        self._buffer[122:162] = struct.pack('<40s', value.encode('ascii'))

    @property
    def start_time(self) -> datetime:
        """Start time of image stream in UTC."""
        timestamp_utc = struct.unpack('<q', self._buffer[170:178])[0]
        return datetime_from_timestamp_utc(timestamp_utc)

    @start_time.setter
    def start_time(self, value: datetime) -> None:
        """Set start time using a timezone-aware datetime object.

        Note that this will set both the UTC and local start time fields in the header such that
        they are consistent.
        """
        timestamp_utc = make_timestamp_utc(value)
        timestamp_local = int(
            timestamp_utc + value.utcoffset().total_seconds() * VB_DATE_TICKS_PER_SEC
        )
        self._buffer[162:170] = struct.pack('<q', timestamp_local)  # local timestamp
        self._buffer[170:178] = struct.pack('<q', timestamp_utc)  # UTC timestamp

    @property
    def start_time_local(self) -> datetime:
        """Start time of image stream in local time."""
        timestamp_local = struct.unpack('<q', self._buffer[162:170])[0]
        timestamp_utc = struct.unpack('<q', self._buffer[170:178])[0]
        # no bounds check here -- assume the UTC offset is reasonable
        utc_offset = timedelta(
            seconds=(timestamp_local - timestamp_utc) / VB_DATE_TICKS_PER_SEC
        )
        return self.start_time.astimezone(timezone(utc_offset))

    @property
    def bytes_per_pixel(self) -> int:
        """Number of bytes used to store each pixel value, a function of bit depth. Read only."""
        return (self.bit_depth - 1) // 8 + 1

    @property
    def frame_size_bytes(self) -> int:
        """Size of a frame in bytes. Read only."""
        frame_size_bytes = self.frame_width * self.frame_height * self.bytes_per_pixel
        if self.color_id in [SERColorID.BGR, SERColorID.RGB]:
            frame_size_bytes *= 3
        return frame_size_bytes

    @property
    def frame_size_pixels(self) -> int:
        """Number of pixels in a frame."""
        return self.frame_width * self.frame_height


class SERReader:
    """Reads a SER file."""

    def __init__(self, filename: str):
        """Construct a SERReader object.

        Args:
            filename: Filename of SER file to read.
        """

        self.file = open(filename, 'rb')

        # read in the header
        self._header = SERHeader()
        self._header.buffer = self.file.read(SERHeader.SIZE_BYTES)

        # numpy dtype for pixel values
        self.dtype = np.dtype(
            self._header.endianness.dtype_char() + f'u{self._header.bytes_per_pixel}'
        )

        # check if trailer is present and read it if so
        self.file.seek(0, SEEK_END)
        file_size_bytes = self.file.tell()
        file_size_no_trailer = (
            SERHeader.SIZE_BYTES +
            self._header.frame_count * self._header.frame_size_bytes
        )
        trailer_size_bytes = 8 * self._header.frame_count  # each timestamp is an int64
        file_size_with_trailer = file_size_no_trailer + trailer_size_bytes
        if file_size_bytes == file_size_no_trailer:
            self.has_trailer = False
        elif file_size_bytes == file_size_with_trailer:
            self.has_trailer = True
            self.file.seek(file_size_no_trailer, SEEK_SET)
            self.timestamps = np.fromfile(self.file, dtype=np.int64, count=self._header.frame_count)
        else:
            raise RuntimeError('Invalid file size!')

    @property
    def header(self) -> SERHeader:
        """Get a copy of the file header."""
        # Return a deep copy to prevent caller from accidentally modifying the contents of this
        # object's copy of the header. The internal copy is read-only.
        return deepcopy(self._header)

    def get_frame(self, frame_index: int) -> np.ndarray:
        """Get a frame.

        Args:
            frame_index: Select which frame to get. First frame has index 0.

        Returns:
            The frame data as a numpy array.
        """

        if frame_index < 0 or frame_index >= self._header.frame_count:
            raise ValueError('Invalid frame index')

        if self._header.color_id in [SERColorID.BGR, SERColorID.RGB]:
            raise NotImplementedError('Not implemented yet for BGR or RGB')

        self.file.seek(SERHeader.SIZE_BYTES + frame_index * self._header.frame_size_bytes, SEEK_SET)

        # This won't work for BGR or RGB
        return np.reshape(
            np.fromfile(self.file, dtype=self.dtype, count=self._header.frame_size_pixels),
            (self._header.frame_height, self._header.frame_width)
        )

    def get_frame_vb_timestamp(self, frame_index: int) -> int:
        """Get the raw VB frame timestamp if trailer is present.

        Args:
            frame_index: Index of the frame for which a timestamp is desired.

        Returns:
            The integer VB timestamp value.

        Raises:
            RuntimeError if there is no trailer in this SER file.
        """
        if not self.has_trailer:
            raise RuntimeError('No trailer present in this SER file.')
        return self.timestamps[frame_index]

    def get_frame_timestamp(self, frame_index: int) -> datetime:
        """Get frame timestamp as a datetime object.

        Args:
            frame_index: Index of the frame for which a timestamp is desired.

        Returns:
            A datetime object representing the frame timestamp.

        Raises:
            RuntimeError if there is no trailer in this SER file.
        """
        return datetime_from_timestamp_utc(self.get_frame_vb_timestamp(frame_index))


class SERWriter:
    """Writes a SER file."""

    def __init__(
            self,
            filename: str,
            header: SERHeader,
            add_trailer: bool = True,
        ):
        """Constructs a SERWriter object.

        Args:
            filename: Filename of SER file to create.
            header: `SERHeader` object which defines the header fields to use.
            add_trailer: Trailer containing frame timestamps will be added if
                `True`, otherwise trailer will be omitted.
        """
        self.header = header
        self.header.frame_count = 0
        self.add_trailer = add_trailer
        self.timestamps = []

        self.file = open(filename, 'wb')
        # header will be written on close, seek past it for now
        self.file.seek(SERHeader.SIZE_BYTES, SEEK_SET)

        # numpy dtype for pixel values
        self.dtype = np.dtype(
            self.header.endianness.dtype_char() + f'u{self.header.bytes_per_pixel}'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self) -> None:
        """Write header and trailer (if enabled) and close file."""

        # write the header
        self.file.seek(0, SEEK_SET)
        self.file.write(self.header.buffer)

        # write the trailer
        if self.add_trailer:
            self.file.seek(0, SEEK_END)
            self.file.write(struct.pack(f'<{self.header.frame_count}q', *self.timestamps))

        self.file.close()

    def add_frame(
            self,
            frame: np.ndarray,
            timestamp: Optional[Union[datetime, int]] = None,
        ) -> None:
        """Add new frame to the file.

        The new frame will be appended to the file behind any frames added previously.

        Args:
            frame: A numpy array with the frame data to be written.
            timestamp: Timestamp to store with this frame if trailer is enabled. If None, the
                current time will be used to generate the timestamp. Otherwise, provide either a
                datetime object or the encoded integer VB timestamp value which will be written
                unmodified to the SER file trailer.
        """
        if frame.shape != (self.header.frame_height, self.header.frame_width):
            raise ValueError('Incorrect frame shape')

        if self.header.color_id in [SERColorID.BGR, SERColorID.RGB]:
            raise NotImplementedError('Not implemented yet for BGR or RGB')

        frame.tofile(self.file)
        self.header.frame_count += 1

        if timestamp is None:
            timestamp = make_timestamp_utc(datetime.utcnow())
        elif isinstance(timestamp, datetime):
            timestamp = make_timestamp_utc(timestamp)
        self.timestamps.append(timestamp)
