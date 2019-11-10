"""GPS location determination via gpsd.

Defines a single class GPS that encapsulates the procedures necessary to obtain the current location
from the gpsd daemon. This code works on the assumption that gpsd is already configured and running
on the system.
"""

# http://manpages.ubuntu.com/manpages/bionic/man5/gpsd_json.5.html

from collections import namedtuple
from enum import Flag, auto
import math
import time
#from types import SimpleNamespace

import gps

class GPS:
    """Class encapsulating code for interfacing with gpsd.

    TODO: add more docstring crap everywhere to satisfy the Brett
    """

    class FailureReason(Flag):
        NONE      = 0
        BAD_FIX   = auto() # insufficient fix type
        NO_LAT    = auto() # no latitude
        NO_LON    = auto() # no longitude
        ERR_TIME  = auto() # excessive error in this parameter
        ERR_LAT   = auto() # excessive error in this parameter
        ERR_LON   = auto() # excessive error in this parameter
        ERR_ALT   = auto() # excessive error in this parameter
        ERR_TRACK = auto() # excessive error in this parameter
        ERR_SPEED = auto() # excessive error in this parameter
        ERR_CLIMB = auto() # excessive error in this parameter

    class GetLocationFailure(Exception):
        def __init__(self, reason):
            super().__init__('failure reason bits: {}'.format(reason))
            self.reason = reason

    Errors = namedtuple('Errors', ['time', 'lat', 'lon', 'alt', 'track', 'speed', 'climb'])

    Location = namedtuple('Location', ['lat', 'lon', 'alt'])

    def __init__(self, host='::1', port=gps.GPSD_PORT):
        self.client = gps.gps(host, port, verbose=0, mode=gps.WATCH_ENABLE, reconnect=True)

        self.fix_type = gps.MODE_NO_FIX
        self.location = self.Location(math.nan, math.nan, 0.0)
        self.errors   = self.Errors(math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf)

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
    # on success: returns a Location tuple
    # on failure: raises GetLocationFailure with flags showing which requirements were not met
    # parameters:
    # - timeout: how much time to spend attempting to get a good fix before giving up
    #            (set to None for no timeout, i.e. try forever)
    # - need_3d: whether a 3D fix should be considered necessary
    # - err_max: an Errors tuple containing the thresholds that each 95% error value must be below
    #            for the fix to be considered satisfactory
    #            (individual members of the tuple may be set to math.inf to ignore those ones)
    # TODO: we *may* (not sure!) need to ensure that we get N consecutive reports that all meet our
    #       requirements, before we declare ourselves successful and victorious
    def get_location(self, timeout, need_3d, err_max):
        if timeout is not None:
            assert timeout > 0.0
            t_start = time.perf_counter()

        for report in self.client:
            if report['class'] != 'TPV': continue

            if 'mode' in report and report.mode != 0:
                self.fix_type = report.mode

            new_location = self.location._asdict() # TODO: make sure this works properly
            if 'lat' in report: new_location['lat'] = report.lat
            if 'lon' in report: new_location['lon'] = report.lon
            if 'alt' in report: new_location['alt'] = report.alt
            self.location = self.Location(**new_location) # TODO: make sure this works properly

            new_errors = self.errors._asdict() # TODO: make sure this works properly
            if 'ept' in report: new_errors['time']  = report.ept
            if 'epy' in report: new_errors['lat']   = report.epy
            if 'epx' in report: new_errors['lon']   = report.epx
            if 'epv' in report: new_errors['alt']   = report.epv
            if 'epd' in report: new_errors['track'] = report.epd
            if 'eps' in report: new_errors['speed'] = report.eps
            if 'epc' in report: new_errors['climb'] = report.epc
            self.errors = self.Errors(**new_errors) # TODO: make sure this works properly

            if self._satisfies_criteria(need_3d, err_max):
                return self.location

            if timeout is not None:
                t_now = time.perf_counter()
                if t_now - t_start >= timeout:
                    self._raise_failure(need_3d, err_max)

    def _satisfies_criteria(self, need_3d, err_max):
        if             self.fix_type < gps.MODE_2D: return False
        if need_3d and self.fix_type < gps.MODE_3D: return False

        if math.isnan(self.location.lat): return False
        if math.isnan(self.location.lon): return False

        # TODO: make sure this works properly
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]: return False

        return True

    def _raise_failure(self, need_3d, err_max):
        reason = self.FailureReason.NONE

        if             self.fix_type < gps.MODE_2D: reason |= self.FailureReason.BAD_FIX
        if need_3d and self.fix_type < gps.MODE_3D: reason |= self.FailureReason.BAD_FIX

        if math.isnan(self.location.lat): reason |= self.FailureReason.NO_LAT
        if math.isnan(self.location.lon): reason |= self.FailureReason.NO_LON

        # TODO: make sure this works properly
        for field in self.errors._fields:
            if self.errors._asdict()[field] > err_max._asdict()[field]:
                # TODO: make sure this works properly
                reason |= next(val for (name, val) in self.FailureReason.__members__.items() if name == 'ERR_' + field.upper())

        raise self.GetLocationFailure(reason)
