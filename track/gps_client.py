#!/usr/bin/env python3

"""GPS location determination via gpsd.

Defines a single class GPS that encapsulates the procedures necessary to obtain the current
location as a client of the gpsd daemon.

The code in this module works on the assumption that gpsd is already configured and running on the
system.

It is also highly suggested that ntpd is also already configured and running on the system, and
preferably is configured to use the gps module's pps signal via gpsd as its time source. In any
case, the validation code in this module will (optionally) enforce that the gps time matches the
system time to a reasonable degree of precision.
"""

# http://manpages.ubuntu.com/manpages/bionic/man5/gpsd_json.5.html

import logging
from typing import NamedTuple
from datetime import datetime
from enum import Flag, IntEnum, auto
from math import inf, isinf, nan, isnan
from time import perf_counter
import click
from configargparse import Namespace
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time as APTime
import gps
from track.config import ArgParser


logger = logging.getLogger(__name__)


class GPSFixType(IntEnum):
    """Improved enum-ized version of the 'MODE_' constants from the gps module."""

    ZERO = 0
    NO_FIX = 1
    FIX_2D = 2
    FIX_3D = 3


class GPSValues(NamedTuple):
    """Values provided by the GPS.

    This tuple may be used to store one of two types of information:
    1) The estimates from the GPS.
    2) The 95% confidence intervals corresponding to a set of estimates. Always non-negative. Each
        attribute in this flavor of the tuple has the same unit as the corresponding estimate
        value.

    Attributes:
        lat: Latitude of the GPS receiver in degrees. Range: [-90, +90], positive North.
        lon: Longitude of the GPS receiver in degrees. Range: [0,360), positive East.
        alt: Altitude of the GPS receiver in meters. Range: (-Inf, +Inf).
        track: Standard compass heading in degrees. Range: [0,360), clockwise from North.
        speed: Speed in the XY plane in meters per second. Range: [0.0, +Inf).
        climb: Rate of ascent or decent in meters per second. Range: (-Inf, +Inf); upward positive.
        time: Absolute UTC time as string in RFC 3339 format; up to millisecond precision.
    """

    lat: float
    lon: float
    alt: float
    track: float
    speed: float
    climb: float
    time: str


class GPSMargins(NamedTuple):
    """Allowed margins between GPS-estimated values and some truth reference for sanity checks.

    For all of the below attributes a value of inf means "don't-care."

    Attributes:
        speed: max allowed deviation from zero for horizontal speed in meters per second.
        climb: max allowed deviation from zero for vertical speed in meters per second.
        time: max allowed deviation between gps time and system time in seconds.
    """

    speed: float
    climb: float
    time: float


class GPS:
    """Class encapsulating code for interfacing with gpsd."""

    INIT_FIX = GPSFixType.NO_FIX
    INIT_VAL = GPSValues(lat=nan, lon=nan, alt=nan, track=nan, speed=nan, climb=nan, time=None)
    INIT_ERR = GPSValues(lat=inf, lon=inf, alt=inf, track=inf, speed=inf, climb=inf, time=inf)

    class FailureReason(Flag):
        """Various reasons a GPS request might fail."""

        NONE = 0
        BAD_FIX = auto()  # insufficient fix type
        NO_LAT = auto()  # no latitude
        NO_LON = auto()  # no longitude
        NO_ALT = auto()  # no altitude (only if 3D fix is required)
        BAD_SPEED = auto()  # horizontal speed != zero
        BAD_CLIMB = auto()  # vertical speed != zero
        BAD_TIME = auto()  # gps time != system time (suggests a ntpd<-->gpsd sync failure)
        ERR_LAT = auto()  # excessive error in parameter: lat
        ERR_LON = auto()  # excessive error in parameter: lon
        ERR_ALT = auto()  # excessive error in parameter: alt
        ERR_TRACK = auto()  # excessive error in parameter: track
        ERR_SPEED = auto()  # excessive error in parameter: speed
        ERR_CLIMB = auto()  # excessive error in parameter: climb
        ERR_TIME = auto()  # excessive error in parameter: time

    class GetLocationFailure(Exception):
        """Raised by get_location() when timeout expires before success criteria are met.

        When you get this exception:
        - Look at the flags set in the exception's reason field to see which things did not pass.
            If no flags are set, this means no reports were received from gpsd before the timeout
            expired.
        - Look at the GPS object's fix_type, values, and errors fields to get exact information.
        """

        def __init__(self, reason: "GPS.FailureReason"):
            super().__init__(str(reason))
            self.reason = reason

    def __init__(self, host: str = '::1', port: str = gps.GPSD_PORT):
        """Construct an instance of GPS.

        Creates a client TCP connection to gpsd.

        Args:
            host: Hostname or IP address of machine running gpsd.
            port: A string giving the port number on which gpsd listens for clients (usually 2947).
        """
        self.client = gps.gps(host, port, verbose=0, mode=gps.WATCH_ENABLE, reconnect=True)

        self.fix_type = self.INIT_FIX
        self.values = self.INIT_VAL
        self.errors = self.INIT_ERR

    def __enter__(self):
        """Support usage of this class in 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support usage of this class in 'with' statements."""
        self.close()

    def close(self):
        """Close this client's connection to gpsd."""
        self.client.close()

    def get_location(
        self, timeout: float, need_3d: bool, err_max: GPSValues, margins: GPSMargins
    ) -> EarthLocation:
        """Get location estimate from GPS receiver.

        This method queries the gpsd service for estimates from a GPS receiver until either a
        timeout expires or an estimate meeting certain criteria is received.

        Requirements for successful execution (return with no exception raised):
        - don't time out
        - get a 2D or 3D fix (depending on value of `need_3d` argument)
        - all relevant confidence intervals need to be within the specified bounds

        Args:
            timeout: How much time to spend attempting to get a good fix before giving up, in
                seconds. Set to inf for no timeout (try forever).
            need_3d: True indicates that a 3D fix is required.
            err_max: A GPSValues tuple containing the thresholds that each 95% error value must be
                below for the fix to be considered satisfactory. Individual members of the tuple
                may be set to inf to request no checking.
            margins: A GPSMargins tuple indicating how far from zero the speed and climb values can
                be, and how far from the system time the GPS time can be, for a satisfactory fix.
                Individual members of the tuple may be set to inf to request no checking.

        Returns:
            On success, returns an instance of EarthLocation with the GPS-estimated location.

        Raises:
            GetLocationFailure on failure with flags showing which requirements were not met.
            RuntimeError if there is a bug in the logic allowing escape from the loop.
        """
        if timeout < 0.0 or isnan(timeout):
            raise ValueError()

        if not isinstance(err_max, GPSValues):
            raise TypeError()
        if not isinstance(margins, GPSMargins):
            raise TypeError()

        for v in err_max + margins:
            if not isinstance(v, float):
                raise TypeError()
            if v < 0.0 or isnan(v):
                raise ValueError()

        t_start = perf_counter()

        # read reports until all criteria are satisfied (return) or timeout (exception)
        fail_reasons = self.FailureReason.NONE
        while True:
            if perf_counter() - t_start >= timeout:
                raise self.GetLocationFailure(fail_reasons)

            # check if reports are ready before calling read() to prevent blocking forever
            if not self.client.waiting(timeout=0.01):
                continue
            if self.client.read() == -1:
                continue

            report = self.client.data

            if report['class'] != 'TPV':
                continue

            if 'mode' in report:
                if report.mode != 0:
                    self.fix_type = GPSFixType(report.mode)
                else:
                    self.fix_type = GPSFixType.NO_FIX

            self.values = GPSValues(
                lat=report.lat if 'lat' in report else self.INIT_VAL.lat,
                lon=report.lon if 'lon' in report else self.INIT_VAL.lon,
                alt=report.alt if 'alt' in report else self.INIT_VAL.alt,
                track=report.track if 'track' in report else self.INIT_VAL.track,
                speed=report.speed if 'speed' in report else self.INIT_VAL.speed,
                climb=report.climb if 'climb' in report else self.INIT_VAL.climb,
                time=report.time if 'time' in report else self.INIT_VAL.time,
            )

            if isnan(self.values.alt) and need_3d is False:
                # since a 3D fix is not required, set alt to 0
                self.values = self.values._replace(alt=0.0)

            self.errors = GPSValues(
                lat=report.epy if 'epy' in report else self.INIT_ERR.lat,
                lon=report.epx if 'epx' in report else self.INIT_ERR.lon,
                alt=report.epv if 'epv' in report else self.INIT_ERR.alt,
                track=report.epd if 'epd' in report else self.INIT_ERR.track,
                speed=report.eps if 'eps' in report else self.INIT_ERR.speed,
                climb=report.epc if 'epc' in report else self.INIT_ERR.climb,
                time=report.ept if 'ept' in report else self.INIT_ERR.time,
            )

            fail_reasons = self._check_criteria(need_3d, err_max, margins)
            if fail_reasons == self.FailureReason.NONE:
                return EarthLocation(
                    lat=self.values.lat * u.deg,
                    lon=self.values.lon * u.deg,
                    height=self.values.alt * u.m,
                )

            if perf_counter() - t_start >= timeout:
                raise self.GetLocationFailure(fail_reasons)

        raise RuntimeError('GPS: unexpected code path')

    def _check_criteria(
        self, need_3d: bool, err_max: GPSValues, margins: GPSMargins
    ) -> "GPS.FailureReason":
        """Check that the values provided by the GPS receiver meet certain criteria.

        Args:
            need_3d: If True, a 3D fix will be required. Otherwise, a 2D fix will be accepted.
            err_max: A GPSValues tuple containing the thresholds that each 95% error value must be
                below for the fix to be considered satisfactory (individual members of the tuple
                may be set to inf to request no checking).
            margins: A GPSMargins tuple containing the absolute error margins that apply to certain
                GPS values.

        Returns:
            An instance of FailureReason with flags set to indicate checks that failed.
        """
        fail_reasons = self.FailureReason.NONE

        min_fix = GPSFixType.FIX_3D if need_3d else GPSFixType.FIX_2D
        if self.fix_type < min_fix:
            fail_reasons |= self.FailureReason.BAD_FIX

        if isnan(self.values.lat):
            fail_reasons |= self.FailureReason.NO_LAT
        if isnan(self.values.lon):
            fail_reasons |= self.FailureReason.NO_LON
        if isnan(self.values.alt):
            fail_reasons |= self.FailureReason.NO_ALT

        if _test_margin_zero_fail(self.values.speed, margins.speed):
            fail_reasons |= self.FailureReason.BAD_SPEED
        if _test_margin_zero_fail(self.values.climb, margins.climb):
            fail_reasons |= self.FailureReason.BAD_CLIMB
        if _test_margin_time_fail(self.values.time, margins.time):
            fail_reasons |= self.FailureReason.BAD_TIME

        # Check that confidence intervals are within acceptable limits
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]:
                # iterates over FailureReason flags and adds the first one with a matching name
                # pylint: disable=no-member
                fail_reasons |= next(
                    val
                    for (name, val) in self.FailureReason.__members__.items()
                    if name == 'ERR_' + field.upper()
                )

        return fail_reasons


def _test_margin_zero_fail(v_float, margin):
    """Returns True if the given float value is NOT within margin of zero."""
    if isnan(v_float) and not isinf(margin):
        return True
    return v_float > margin


def _test_margin_time_fail(v_str: str, margin: float):
    """Check if the given RFC 3339 time string is NOT within margin seconds of the system time.

    Args:
        v_str: An RFC 3339 time string.
        margin: Allowed error magnitude between `v_str` time and system time in seconds.

    Returns:
        True when the error between v_str and the system time exceeds `margin` seconds.
    """
    if v_str is None and not isinf(margin):
        return True
    if isinf(margin):
        return False  # short-circuit the time-parsing code if we don't need it

    try:
        t_val = APTime(v_str, format='isot', scale='utc')
    except ValueError:
        return True
    t_val = t_val.to_datetime()
    t_sys = datetime.utcnow()

    dt = abs(t_val - t_sys).total_seconds()
    return dt > margin


def add_program_arguments(
    parser: ArgParser,
    group_description: str = 'Setting all three of these options will override GPS',
) -> None:
    """Add program arguments pertaining to observer location.

    Arguments are added such that the location is obtained from a GPS receiver by default, but the
    user can override this by supplying the location at the command line.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
        group_description: The description for this argument group.
    """
    observer_group = parser.add_argument_group(
        title='Observer Location Options',
        description=group_description,
    )
    observer_group.add_argument(
        '--lat',
        help='latitude of observer (+N)',
        type=float,
    )
    observer_group.add_argument(
        '--lon',
        help='longitude of observer (+E)',
        type=float,
    )
    observer_group.add_argument('--elevation', help='elevation of observer (m)', type=float)


def make_location_from_args(args: Namespace) -> EarthLocation:
    """Generate an observer location based on the program arguments provided.

    Args:
        args: Set of program arguments.

    Returns:
        An EarthLocation object.

    Raises:
        ValueError if an invalid combination of program arguments is specified.
        RuntimeError if location is specified by program args but user does not confirm.
    """
    if all(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        print('Location (lat, lon, elevation) specified by program args. This will override GPS.')
        if not click.confirm('Proceed without GPS location?', default=True):
            raise RuntimeError('Could not generate a location')
        location = EarthLocation(
            lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.elevation * u.m
        )
        logger.info('Got location from program args.')
    elif any(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        raise ValueError("Must give all of lat, lon, and elevation or none of them.")
    else:
        with GPS() as g:
            location = g.get_location(
                timeout=10.0,
                need_3d=True,
                err_max=GPSValues(
                    lat=100.0, lon=100.0, alt=100.0, track=inf, speed=inf, climb=inf, time=0.01
                ),
                margins=GPSMargins(speed=inf, climb=inf, time=1.0),
            )
            logger.info('Got location from GPS.')

    logger.info(
        'Generated the following location: '
        f'lat: {location.lat:.5f}, '
        f'lon: {location.lon:.5f}, '
        f'altitude: {location.height:.2f}.'
    )

    return location


def main():
    """Get a GPS reading and print to the console.

    Note that the ERR_MAX and MARGINS values in this function are extremely lax, allowing almost
    any 3D fix to pass muster. Adjust these to be more restrictive as desired.
    """
    # parameters
    timeout = 10.0
    need_3d = True
    err_max = GPSValues(lat=100.0, lon=100.0, alt=inf, track=inf, speed=inf, climb=inf, time=100.0)
    margins = GPSMargins(speed=inf, climb=inf, time=inf)

    with GPS() as g:
        try:
            loc = g.get_location(timeout, need_3d, err_max, margins)
            print(f'lat: {loc.lat:.5f}, lon: {loc.lon:.5f}, altitude: {loc.height:.2f}')
        finally:
            print(f'fix_type: {g.fix_type}')
            print(f'values:   {g.values}')
            print(f'errors:   {g.errors}')


if __name__ == '__main__':
    main()
