"""targets for use in telescope tracking control loop"""

from typing import Tuple
import time
from abc import ABC, abstractmethod
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Longitude
from astropy.time import Time
from astropy import units as u
import ephem
from track.model import MountModel
from track.mounts import MountEncoderPositions


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


class AcceleratingMountAxisTarget(Target):
    """A target that accelerates at a constant rate in one or both mount axes.

    This target is intented for testing the control system's ability to track an accelerating
    target with reasonably small steady-state error.
    """

    def __init__(
            self,
            mount_model: MountModel,
            initial_encoder_positions: MountEncoderPositions,
            axis_accelerations: Tuple[float, float],
        ):
        """Construct an AcceleratingMountAxisTarget.

        Initial velocity of the target is zero in both axes. Acceleration begins the moment this
        constructor is called and continues forever without limit as currently implemented.

        Args:
            mount_model: An instance of the mount model. This is required because the acceleration
                is meant to be held constant in the mount axes, yet the API of the Target class
                requires a topocentric coordinate be returned by the `get_position()` method.
            initial_encoder_positions: The starting positions of the mount encoders. Note that if
                axis 1 starts out pointed at the pole and has acceleration 0 axis 0 may not behave
                as expected since the pole is a singularity. A work-round is to set the initial
                encoder position for axis 1 to a small offset from the pole, or to use a non-zero
                acceleration for axis 1.
            axis_accelerations: A tuple of two floats giving the accelerations in axis 0 and axis
                1, respectively, in degrees per second squared (negative okay).
        """
        self.mount_model = mount_model
        self.accel = axis_accelerations
        self.time_start = None
        self.initial_positions = initial_encoder_positions

    def get_position(self, t: Time):
        """Gets the position of the simulated target for a specific time."""

        if self.time_start is None:
            # Don't do this in constructor because it may be a couple seconds between when the
            # constructor is called until the first call to this method.
            self.time_start = t

        time_elapsed = (t - self.time_start).sec
        target_encoder_positions = MountEncoderPositions(
            Longitude((self.initial_positions[0].deg + self.accel[0] * time_elapsed**2) * u.deg),
            Longitude((self.initial_positions[1].deg + self.accel[1] * time_elapsed**2) * u.deg),
        )
        return self.mount_model.encoders_to_topocentric(target_encoder_positions)


class PyEphemTarget(Target):
    """A target using the PyEphem package"""

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
        self.observer.lat = location.lat.rad
        self.observer.lon = location.lon.rad
        self.observer.elevation = location.height.to_value(u.m)


    def get_position(self, t: Time) -> SkyCoord:
        """Get apparent topocentric position of this target"""
        self.observer.date = ephem.Date(t.datetime)
        self.target.compute(self.observer)
        return SkyCoord(self.target.az * u.rad, self.target.alt * u.rad, frame='altaz')
