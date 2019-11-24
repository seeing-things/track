"""GPS location determination via gpsd.

Defines a single class GPS that encapsulates the procedures necessary to obtain the current location
from the gpsd daemon. This code works on the assumption that gpsd is already configured and running
on the system.
"""

# http://manpages.ubuntu.com/manpages/bionic/man5/gpsd_json.5.html

from collections import namedtuple
from enum        import Enum, Flag, auto
from math        import inf, nan, isnan
from time        import perf_counter
#from types import SimpleNamespace

from astropy import units as u
from astropy.coordinates import EarthLocation

import gps

# improved enum-ized version of the 'MODE_' constants from the gps module
class GPSFixType(Enum):
    ZERO   = 0
    NO_FIX = 1
    FIX_2D = 2
    FIX_3D = 3

GPSLocation = namedtuple('GPSLocation', ['lat', 'lon', 'alt'])
GPSErrors   = namedtuple('GPSErrors',   ['lat', 'lon', 'alt', 'track', 'speed', 'climb', 'time'])

# TODO: document the units on all the relevant fields here, and use types for them when appropriate
#       (ask Brett for the best degree types and unit stuff to use here... astropy has good ones?)
# - lat: decimal degrees, [-90.0, +90.0]; north positive
# - lon: decimal degrees, [-180.0, +180.0] or similar (I think); east positive
# - alt: meters above sea level, [-inf, +inf]
# - track: decimal degrees, [0.0, 360.0); standard compass heading: clockwise from north
# - speed: meters per second in XY plane, [0.0, +inf]
# - climb: meters per second in Z  plane, [-inf, +inf]; upward positive
# - time: absolute UTC time; up to millisecond precision
# - errors: same units as the corresponding value; 95% confidence; always nonnegative;
#           lat/lon errors are in meters
#           time error is in seconds

# TODO: put this somewhere else where it would belong, or find an existing module that provides this
def const(val): return property(fget=lambda _: val)

class GPS:
    """Class encapsulating code for interfacing with gpsd.

    TODO: add more docstring crap everywhere to satisfy the Brett
    """

    INIT_FIX = const(GPSFixType.NO_FIX)
    INIT_LOC = const(GPSLocation(lat=nan, lon=nan, alt=0.0))
    INIT_ERR = const(GPSErrors(lat=inf, lon=inf, alt=inf, track=inf, speed=inf, climb=inf, time=inf))

    class FailureReason(Flag):
        NONE      = 0
        BAD_FIX   = auto() # insufficient fix type
        NO_LAT    = auto() # no latitude
        NO_LON    = auto() # no longitude
        NO_ALT    = auto() # no altitude (currently not actually enforced)
        ERR_LAT   = auto() # excessive error in parameter: lat
        ERR_LON   = auto() # excessive error in parameter: lon
        ERR_ALT   = auto() # excessive error in parameter: alt
        ERR_TRACK = auto() # excessive error in parameter: track
        ERR_SPEED = auto() # excessive error in parameter: speed
        ERR_CLIMB = auto() # excessive error in parameter: climb
        ERR_TIME  = auto() # excessive error in parameter: time

    # when you get this exception:
    # - look at the flags set in the exception's reason field to see which things did not pass
    # - look at the GPS object's fix_type, location, and errors fields to get exact information
    class GetLocationFailure(Exception):
        def __init__(self, reason):
            super().__init__(str(reason))
            self.reason = reason

    def __init__(self, host='::1', port=gps.GPSD_PORT):
        self.client = gps.gps(host, port, verbose=0, mode=gps.WATCH_ENABLE, reconnect=True)

        self.fix_type = self.INIT_FIX
        self.location = self.INIT_LOC
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
    # parameters:
    # - timeout: how much time to spend attempting to get a good fix before giving up
    #            (set to None for no timeout, i.e. try forever)
    # - need_3d: whether a 3D fix should be considered necessary
    # - err_max: an Errors tuple containing the thresholds that each 95% error value must be below
    #            for the fix to be considered satisfactory
    #            (individual members of the tuple may be set to inf to ignore those ones)
    # TODO: we *may* (not sure!) need to ensure that we get N consecutive reports that all meet our
    #       requirements, before we declare ourselves successful and victorious
    # TODO: determine whether we should enforce alt, and in which circumstances
    #       (e.g. should we only enforce it if need_3d is True?)
    #       enforcement would mean:
    #       - change INIT_ALT from 0.0 to nan
    #       - if isnan(self.location.alt) in _satisfies_criteria: return False
    #       - if isnan(self.location.alt) in _raise_failure: add flag NO_ALT
    #       also: if we are doing strictly a 2D fix, should we force self.location.alt to 0.0?
    #       and do we even need to do that, or will the gps hw/sw do that for us...?
    def get_location(self, timeout, need_3d, err_max):
        if timeout is not None:
            assert timeout > 0.0
            t_start = perf_counter()

        if need_3d:
            fix_ok = lambda fix: fix == GPSFixType.FIX_3D
        else:
            fix_ok = lambda fix: fix in (GPSFixType.FIX_3D, GPSFixType.FIX_2D)

        for report in self.client:
            if report['class'] != 'TPV': continue

            if 'mode' in report:
                if report.mode != 0: self.fix_type = GPSFixType(report.mode)
                else:                self.fix_type = GPSFixType.NO_FIX

            self.location = GPSLocation(
                lat = report.lat if 'lat' in report else self.INIT_LOC.lat,
                lon = report.lon if 'lon' in report else self.INIT_LOC.lon,
                alt = report.alt if 'alt' in report else self.INIT_LOC.alt,
            )

            self.errors = GPSErrors(
                lat   = report.epy if 'epy' in report else self.INIT_ERR.lat,
                lon   = report.epx if 'epx' in report else self.INIT_ERR.lon,
                alt   = report.epv if 'epv' in report else self.INIT_ERR.alt,
                track = report.epd if 'epd' in report else self.INIT_ERR.track,
                speed = report.eps if 'eps' in report else self.INIT_ERR.speed,
                climb = report.epc if 'epc' in report else self.INIT_ERR.climb,
                time  = report.ept if 'ept' in report else self.INIT_ERR.time,
            )

            if self._satisfies_criteria(err_max, fix_ok):
                return EarthLocation(
                    lat    = self.location.lat*u.deg,
                    lon    = self.location.lon*u.deg,
                    height = self.location.alt*u.m,
                )

            if timeout is not None:
                t_now = perf_counter()
                if t_now - t_start >= timeout:
                    self._raise_failure(err_max, fix_ok)

    def _satisfies_criteria(self, err_max, fix_ok):
        if not fix_ok(self.fix_type): return False

        if isnan(self.location.lat): return False
        if isnan(self.location.lon): return False

        # TODO: make sure this works properly
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]: return False

        return True

    def _raise_failure(self, err_max, fix_ok):
        reason = self.FailureReason.NONE

        if not fix_ok(self.fix_type): reason |= self.FailureReason.BAD_FIX

        if isnan(self.location.lat): reason |= self.FailureReason.NO_LAT
        if isnan(self.location.lon): reason |= self.FailureReason.NO_LON

        # TODO: make sure this works properly
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]:
                # TODO: make sure this works properly
                reason |= next(val for (name, val) in self.FailureReason.__members__.items() if name == 'ERR_' + field.upper())

        raise self.GetLocationFailure(reason)
