"""GPS location determination via gpsd.

Defines a single class GPS that encapsulates the procedures necessary to obtain the current location
as a client of the gpsd daemon.

The code in this module works on the assumption that gpsd is already configured and running on the
system.

It is also highly suggested that ntpd is also already configured and running on the system, and
preferably is configured to use the gps module's pps signal via gpsd as its time source. In any
case, the validation code in this module will (optionally) enforce that the gps time matches the
system time to a reasonable degree of precision.
"""

# http://manpages.ubuntu.com/manpages/bionic/man5/gpsd_json.5.html

from collections import namedtuple
from datetime    import datetime
from enum        import Flag, IntEnum, auto
from math        import inf, isinf, nan, isnan
from time        import perf_counter
#from types       import SimpleNamespace

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time as APTime

import gps

# improved enum-ized version of the 'MODE_' constants from the gps module
class GPSFixType(IntEnum):
    ZERO   = 0
    NO_FIX = 1
    FIX_2D = 2
    FIX_3D = 3

# TODO: use alternative type-hint syntax for these namedtuple declarations:
#   from typing import NamedTuple
#   class GPSValues(NamedTuple):
#     lat: Angle # or whatever type we actually want to use for this
#     lon: Angle # likewise
#     # etc...
GPSValues  = namedtuple('GPSValues',  ['lat', 'lon', 'alt', 'track', 'speed', 'climb', 'time'])
GPSMargins = namedtuple('GPSMargins',                               ['speed', 'climb', 'time'])

# TODO: write some nicer documentation for GPSMargins: (for all: inf means don't-care)
# - speed: float: max valid deviation from zero for horizontal speed   (unit: meters/second)
# - climb: float: max valid deviation from zero for vertical   speed   (unit: meters/second)
# - time:  float: max valid deviation between gps time and system time (unit: seconds)

# TODO: document the units on all the relevant fields here, and use types for them when appropriate
#       (ask Brett for the best degree types and unit stuff to use here... astropy has good ones?)
# - lat: decimal degrees, [-90.0, +90.0]; north positive
# - lon: decimal degrees, [-180.0, +180.0] or similar (I think); east positive
# - alt: meters above sea level, [-inf, +inf]
# - track: decimal degrees, [0.0, 360.0); standard compass heading: clockwise from north
# - speed: meters per second in XY plane, [0.0, +inf]
# - climb: meters per second in Z  plane, [-inf, +inf]; upward positive
# - time: absolute UTC time; up to millisecond precision; represented as string in RFC 3339 format
# errors: same units as the corresponding value; 95% confidence; always nonnegative;
#         lat/lon errors are in meters
#         time error is in seconds

# TODO: put this somewhere else where it would belong, or find an existing module that provides this
def const(val): return property(fget=lambda _: val)

class GPS:
    """Class encapsulating code for interfacing with gpsd.

    TODO: add more docstring crap everywhere to satisfy the Brett
    """

    INIT_FIX = const(GPSFixType.NO_FIX)
    INIT_VAL = const(GPSValues(lat=nan, lon=nan, alt=0.0, track=nan, speed=nan, climb=nan, time=None))
    INIT_ERR = const(GPSValues(lat=inf, lon=inf, alt=inf, track=inf, speed=inf, climb=inf, time=inf))

    class FailureReason(Flag):
        NONE      = 0
        BAD_FIX   = auto() # insufficient fix type
        NO_LAT    = auto() # no latitude
        NO_LON    = auto() # no longitude
        NO_ALT    = auto() # no altitude (currently not actually enforced)
        BAD_SPEED = auto() # horizontal speed != zero
        BAD_CLIMB = auto() # vertical   speed != zero
        BAD_TIME  = auto() # gps time != system time (suggests a ntpd<-->gpsd sync failure)
        ERR_LAT   = auto() # excessive error in parameter: lat
        ERR_LON   = auto() # excessive error in parameter: lon
        ERR_ALT   = auto() # excessive error in parameter: alt
        ERR_TRACK = auto() # excessive error in parameter: track
        ERR_SPEED = auto() # excessive error in parameter: speed
        ERR_CLIMB = auto() # excessive error in parameter: climb
        ERR_TIME  = auto() # excessive error in parameter: time

    # when you get this exception:
    # - look at the flags set in the exception's reason field to see which things did not pass
    # - look at the GPS object's fix_type, values, and errors fields to get exact information
    class GetLocationFailure(Exception):
        def __init__(self, reason):
            super().__init__(str(reason))
            self.reason = reason

    def __init__(self, host='::1', port=gps.GPSD_PORT):
        self.client = gps.gps(host, port, verbose=0, mode=gps.WATCH_ENABLE, reconnect=True)

        self.fix_type = self.INIT_FIX
        self.values   = self.INIT_VAL
        self.errors   = self.INIT_ERR

    def __enter__(self):
        """Support usage of this class in 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support usage of this class in 'with' statements."""
        self.close()

    def close(self):
        self.client.close()

    # this is the method that will be called by alignment code
    # requirements for successful execution:
    # - don't time out
    # - get a 3D fix
    # - all relevant errors need to be within the specified bounds
    # - we don't care about DGPS
    # on success: returns an astropy.coordinates.EarthLocation value
    # on failure: raises GetLocationFailure with flags showing which requirements were not met
    # (also may raise RuntimeError if something screwy happens with gpsd)
    # parameters:
    # - timeout: how much time to spend attempting to get a good fix before giving up
    #            (set to inf for no timeout, i.e. try forever)
    # - need_3d: whether a 3D fix should be considered necessary
    # - err_max: a GPSValues tuple containing the thresholds that each 95% error value must be below
    #            for the fix to be considered satisfactory
    #            (individual members of the tuple may be set to inf to request no checking)
    # - margins: a GPSMargins tuple indicating how far from zero the speed and climb values can be,
    #            and how far from the system time the GPS time can be, for a satisfactory fix
    #            (individual members of the tuple may be set to inf to request no checking)
    # TODO: we *may* (not sure!) need to ensure that we get N consecutive reports that all meet our
    #       requirements, before we declare ourselves successful and victorious
    # TODO: determine whether we should enforce alt, and in which circumstances
    #       (e.g. should we only enforce it if need_3d is True?)
    #       enforcement would mean:
    #       - change INIT_ALT from 0.0 to nan
    #       - if isnan(self.values.alt) in _satisfies_criteria: return False
    #       - if isnan(self.values.alt) in _raise_failure: add flag NO_ALT
    #       also: if we are doing strictly a 2D fix, should we force self.values.alt to 0.0?
    #       and do we even need to do that, or will the gps hw/sw do that for us...?
    def get_location(self, timeout, need_3d, err_max, margins):
        if timeout < 0.0 or isnan(timeout): raise ValueError()

        if not isinstance(err_max, GPSValues):  raise TypeError()
        if not isinstance(margins, GPSMargins): raise TypeError()

        for v in err_max + margins:
            if not isinstance(v, float): raise TypeError()
            if v < 0.0 or isnan(v):      raise ValueError()

        t_start = perf_counter()

        for report in self.client:
            if report['class'] != 'TPV': continue

            if 'mode' in report:
                if report.mode != 0: self.fix_type = GPSFixType(report.mode)
                else:                self.fix_type = GPSFixType.NO_FIX

            self.values = GPSValues(
                lat   = report.lat   if 'lat'   in report else self.INIT_VAL.lat,
                lon   = report.lon   if 'lon'   in report else self.INIT_VAL.lon,
                alt   = report.alt   if 'alt'   in report else self.INIT_VAL.alt,
                track = report.track if 'track' in report else self.INIT_VAL.track,
                speed = report.speed if 'speed' in report else self.INIT_VAL.speed,
                climb = report.climb if 'climb' in report else self.INIT_VAL.climb,
                time  = report.time  if 'time'  in report else self.INIT_VAL.time,
            )

            self.errors = GPSValues(
                lat   = report.epy if 'epy' in report else self.INIT_ERR.lat,
                lon   = report.epx if 'epx' in report else self.INIT_ERR.lon,
                alt   = report.epv if 'epv' in report else self.INIT_ERR.alt,
                track = report.epd if 'epd' in report else self.INIT_ERR.track,
                speed = report.eps if 'eps' in report else self.INIT_ERR.speed,
                climb = report.epc if 'epc' in report else self.INIT_ERR.climb,
                time  = report.ept if 'ept' in report else self.INIT_ERR.time,
            )

            fail = self._check_criteria(need_3d, err_max, margins)
            if fail == self.FailureReason.NONE:
                return EarthLocation(
                    lat    = self.values.lat*u.deg,
                    lon    = self.values.lon*u.deg,
                    height = self.values.alt*u.m,
                )

            if perf_counter() - t_start >= timeout:
                raise self.GetLocationFailure(fail)

        raise RuntimeError('gpsd client unexpectedly stopped receiving reports somehow')

    def _check_criteria(self, need_3d, err_max, margins):
        fail = self.FailureReason.NONE

        min_fix = (GPSFixType.FIX_3D if need_3d else GPSFixType.FIX_2D)
        if self.fix_type < min_fix: fail |= self.FailureReason.BAD_FIX

        if isnan(self.values.lat): fail |= self.FailureReason.NO_LAT
        if isnan(self.values.lon): fail |= self.FailureReason.NO_LON
        # TODO: possibly enforce altitude checks

        if _test_margin_zero_fail(self.values.speed, margins.speed): fail |= self.FailureReason.BAD_SPEED
        if _test_margin_zero_fail(self.values.climb, margins.climb): fail |= self.FailureReason.BAD_CLIMB
        if _test_margin_time_fail(self.values.time,  margins.time):  fail |= self.FailureReason.BAD_TIME

        # TODO: make sure this works properly
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]:
                # TODO: make sure this works properly
                # TODO: add a comment here for mere mortals, explaining how the hell this line of code works
                fail |= next(val for (name, val) in self.FailureReason.__members__.items() if name == 'ERR_' + field.upper())

        return fail

# TODO: decide whether margin tests should succeed or fail when the speed/climb/time value is
#       unset (nan/None)... perhaps only if the margin comparison value is not inf? ugh...

_TEST_MARGIN_FAIL_IF_UNSET = False

# returns True if the given float value is NOT within margin of zero
def _test_margin_zero_fail(v_float, margin):
    if isnan(v_float): return _TEST_MARGIN_FAIL_IF_UNSET
    return v_float > margin

# returns True if the given RFC 3339 time string is NOT within margin seconds of the system time
def _test_margin_time_fail(v_str, margin):
    if v_str is None: return _TEST_MARGIN_FAIL_IF_UNSET
    if isinf(margin): return False # short-circuit the time-parsing code if we don't need it

    try:
        t_val = APTime(v_str, format='isot', scale='utc')
    except ValueError:
        return _TEST_MARGIN_FAIL_IF_UNSET
    t_val = t_val.to_datetime()
    t_sys = datetime.utcnow()

    dt = abs(t_val - t_sys).total_seconds()
    return dt > margin
