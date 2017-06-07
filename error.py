from track import ErrorSource
import ephem
import datetime
import math

# wraps an angle in degrees to the range [-180,+180)
def wrap_error(e):
    return (e + 180.0) % 360.0 - 180.0

class BlindErrorSource(ErrorSource):

    def __init__(self, mount, observer, target):

        # PyEphem Observer and Body objects
        self.observer = observer
        self.target = target

        # TelescopeMount object
        self.mount = mount

    def compute_error(self):

        # get current coordinates of the target in degrees
        # not using ephem.now() because it rounds time to the nearest second
        self.observer.date = ephem.Date(datetime.datetime.utcnow())
        self.target.compute(self.observer)
        target_az_deg = self.target.az * 180.0 / math.pi
        target_alt_deg = self.target.alt * 180.0 / math.pi

        # get current position of telescope (degrees)
        (scope_az_deg, scope_alt_deg) = self.mount.get_azel()
         
        # compute pointing errors in degrees
        error_az = wrap_error(target_az_deg - scope_az_deg)
        error_alt = wrap_error(target_alt_deg - scope_alt_deg)

        return (error_az, error_alt)


#class OpticalErrorSource(ErrorSource):
