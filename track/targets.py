"""targets for use in telescope tracking control loop"""

from abc import ABC, abstractmethod
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy import units as u
import ephem


class Target(ABC):
    """Abstract base class providing a common interface for targets to be tracked."""

    @abstractmethod
    def get_position(self, t: Time) -> SkyCoord:
        """Get the apparent topocentric position of the target for the specified time.

        Args:
            t: The time for which the position should correspond.

        Returns:
            A SkyCoord object having AltAz frame giving the apparent position of the object in the
            topocentric reference frame for the given time and for the observer's location.
        """


class FixedTopocentricTarget(Target):
    """A target at a fixed topocentric position.

    Targets of this type remain at a fixed apparent position in the sky. An example might be a tall
    building. These objects do not appear to move as the Earth rotates and do not have any
    significant velocity relative to the observer.
    """

    def __init__(self, coord: SkyCoord):
        if not isinstance(coord.frame, AltAz):
            raise TypeError('frame of coord must be AltAz')
        self.coord = coord

    def get_position(self, t: Time = None) -> SkyCoord:
        """Since the topocentric position is fixed the t argument is ignored"""
        return self.coord


class PyEphemTarget(Target):

    def __init__(self, target, location: EarthLocation):
        """Init a PyEphem target

        This target type uses PyEphem, the legacy package for ephemeris calculations.

        Args:
            target: One of the various PyEphem body objects. Objects in this category should have
                a compute() method.
            location: Location of the observer.
        """

        self.target = target

        # Create a PyEphem Observer object for the given location
        self.observer = ephem.Observer()
        self.observer.lat = location.lat.deg
        self.observer.lon = location.lon.deg
        self.observer.elevation = location.height.to_value(u.m)


    def get_position(self, t: Time) -> SkyCoord:
        """Get apparent position of this target"""
        self.observer.date = ephem.Date(t.datetime)
        self.target.compute(self.observer)
        return SkyCoord(self.target.ra * u.rad, self.target.dec * u.rad)
