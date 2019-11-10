"""Mount modeling and transformations between sky and mount reference frames"""

from typing import NamedTuple
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
from astropy.utils import iers
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Latitude, Longitude, Angle
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates.representation import UnitSphericalRepresentation, CartesianRepresentation


# Try to download IERS data if online but disable strict checking in subsequent calls to methods
# that depend on this data if it is a bit stale. Ideally this data should be fresh, so if this code
# is to be used offline it would be best to provide a mechanism for updating Astropy's IERS cache
# whenever network connectivity is available.
# See https://docs.astropy.org/en/stable/utils/iers.html for additional details.
iers.IERS_Auto().open()  # Will try to download if cache is stale
iers.conf.auto_max_age = None  # Disable strict stale cache check


class ModelParameters(NamedTuple):
    """Set of parameters for the mount model.

    When paired with the equations in the world_to_mount and mount_to_world functions these
    parameters define a unique transformation between mount encoder positions and coordinates in
    a local equatorial coordinate system. When further augmented with a location and time, these
    local coordinates can be further transformed to positions on the celestial sphere.

    Attributes:
        lon_axis_offset: Encoder zero-point offset for the longitude mount axis. For equatorial
            mounts this is the right ascension axis. For altazimuth mounts this is the azimuth
            axis.
        lat_axis_offset: Encoder zero-point offset for the latitude mount axis. For equatorial
            mounts this is the declination axis. For altazimuth mounts this is the altitude axis.
        pole_rot_axis_lon: The longitude angle of the axis of rotation used to transform from a
            spherical coordinate system using the mount physical pole to a coordinate system using
            the celestial pole.
        pole_rot_angle: The angular separation between the instrument pole and the celestial pole.
    """
    lon_axis_offset: Angle
    lat_axis_offset: Angle
    pole_rot_axis_lon: Angle
    pole_rot_angle: Angle

    @staticmethod
    def from_ndarray(param_array):
        """Factory method to generate an instance of this class with values given in an ndarray.

        Args:
            param_array (ndarray): An array of parameter values. This format is required when
                interfacing with the scipy least_squares method.
        """
        return ModelParameters(
            lon_axis_offset=Angle(param_array[0]*u.deg),
            lat_axis_offset=Angle(param_array[1]*u.deg),
            pole_rot_axis_lon=Angle(param_array[2]*u.deg),
            pole_rot_angle=Angle(param_array[3]*u.deg),
        )

    def to_ndarray(self):
        """Return an ndarray containing the model parameters.

        The output format is suitable for use with scipy least_squares.

        Returns:
            An ndarray object containing the parameter values.
        """
        return np.array([
            self.lon_axis_offset.deg,
            self.lat_axis_offset.deg,
            self.pole_rot_axis_lon.deg,
            self.pole_rot_angle.deg,
        ])


def offset_lonlat(coord, lon_offset, lat_offset):
    """Applies offsets to a pair of spherical coordinates.

    This function is mainly required to handle cases where the offsets applied cause the latitude
    angle to go outside the range [-90, +90]. When this occurs, the direction vector crosses "over
    the pole" (say, from +89 to +91 degrees), which requires it to be wrapped to +89 degrees and
    the longitude angle must offset by 180 degrees.

    Args:
        coord (astropy UnitSphericalRepresentation): Coordinate to be adjusted.
        lon_offset (astropy.coordinates.Angle): Offset to be applied to the longitude angle.
        lat_offset (astropy.coordinates.Angle): Offset to be applied to the latitude angle.

    Returns:
        astropy UnitSphericalRepresentation with offsets applied.
    """

    lon = coord.lon + lon_offset
    lat = coord.lat + lat_offset

    # Handle latitude angles outside of [-90, +90]. This is not as simple as a typical modulo
    # wrapping operation because angles that go just beyond one end of the range to not jump to
    # the other extreme; rather they continue back down from the same end of the range but in the
    # opposite direction.
    lat = (lat + 90*u.deg) % (360*u.deg)
    if lat > 180*u.deg:
        lat = 360*u.deg - lat
        # since latitude crossed the pole an odd number of times need to flip longitude by 180
        lon += 180.0*u.deg
    lat -= 90*u.deg

    return UnitSphericalRepresentation(lon, lat)


def tip_axis(coord, axis_lon, rot_angle):
    """Perform a rotation about an axis perpendicular to the Z-axis.

    The purpose of this rotation is to move the pole of the coordinate system from one place to
    another. For example, transforming from a coordinate system where the pole is aligned with the
    physical pole of a mount to a celestial coordinate system where the pole is aligned with the
    celestial pole.

    Note that this is a true rotation, and the same rotation is applied to all coordinates in the
    originating coordinate system equally. It is not equivalent to using SkyCoord
    directional_offset_by() with a fixed position angle and separation since the direction and
    magntidue of the offset depend on the value of coord.

    Args:
        coord (UnitSphericalRepresentation): Coordinate to be transformed.
        axis_lon (Angle): Longitude angle of the axis of rotation.
        rot_angle (Angle): Angle of rotation.

    Returns:
        UnitSphericalRepresentation of the coordinate after rotation is applied.
    """
    rot = rotation_matrix(rot_angle, axis=SkyCoord(axis_lon, 0*u.deg).represent_as('cartesian').xyz)
    coord_cart = coord.represent_as(CartesianRepresentation)
    coord_rot_cart = coord_cart.transform(rot)
    return coord_rot_cart.represent_as(UnitSphericalRepresentation)


def mount_to_world(
        mount_coord,
        t,
        model_params,
    ):
    """Convert coordinate in mount frame to coordinate in celestial equatorial frame.

    Args:
        mount_coord (UnitSphericalRepresentation): Coordinate in mount frame.
        t (Time): An Astropy Time object. This must be initialized with a location as well as time.
        model_params (ModelParameters): The parameter set to use in this transformation.

    Returns:
        A SkyCoord object with the right ascension and declination coordinates in the celestial
            coordinate system.
    """

    # apply encoder offsets
    us_mnt_offset = offset_lonlat(
        mount_coord,
        lon_offset=-model_params.lon_axis_offset,
        lat_offset=-model_params.lat_axis_offset,
    )

    # transform from mount pole to celestial pole
    us_local = tip_axis(
        us_mnt_offset,
        model_params.pole_rot_axis_lon,
        -model_params.pole_rot_angle
    )

    # Calculate the right ascension at this time and place corresponding to the local hour angle.
    ra = t.sidereal_time('mean') - us_local.lon

    return SkyCoord(ra, us_local.lat, frame='icrs')


def world_to_mount(
        sky_coord,
        t,
        model_params,
    ):
    """Convert coordinate in celestial equatorial frame to coordinate in mount frame.

    Args:
        sky_coord: SkyCoord object to be converted to equivalent mount encoder positions.
        t (Time): An Astropy Time object. This must be initialized with a location as well as time.
        model_params (ModelParameters): The parameter set to use in this transformation.

    Returns:
        UnitSphericalRepresentation object with lon corresponding to the right ascension or
        azimuth axis encoder position and lat corresponding to the declination or altitude axis
        encoder position.
    """

    # Calculate hour angle corresponding to SkyCoord right ascension at this time and place
    ha = t.sidereal_time('mean') - sky_coord.ra
    sc_local = SkyCoord(ha, sky_coord.dec)

    # transform from celestial pole to mount pole
    us_mnt_offset = tip_axis(sc_local, model_params.pole_rot_axis_lon, model_params.pole_rot_angle)

    # apply encoder offsets
    mount_coord = offset_lonlat(
        us_mnt_offset,
        model_params.lon_axis_offset,
        model_params.lat_axis_offset
    )

    return mount_coord


def mount_to_losmandy(mount_coord, meridian_side='east'):
    """Convert from mount-relative hour and declination angles to Lonsmandy encoder positions

    Args:
        mount_coord (UnitSphericalRepresentation): Coordinate in the mount frame.
        meridian_side (string): Desired side of mount-relative meridian, 'east' or 'west'. If the
            pole of the mount is not in the direction of the celestial pole this may not
            correspond to true east and west directions.

    Returns:
        A tuple of Longitude objects containing the Losmandy physical encoder positions.
    """
    if meridian_side == 'east':
        pra = Longitude(90*u.deg - mount_coord.lon)
        pdec = Longitude(90*u.deg + mount_coord.lat)
    else:
        pra = Longitude(270*u.deg - mount_coord.lon)
        pdec = Longitude(270*u.deg - mount_coord.lat)
    return pra, pdec


def losmandy_to_mount(pra, pdec):
    """Convert from Losmandy mount encoder positions to mount-relative hour and declination angles

    Args:
        pra (Longitude): Losmandy physical right ascension encoder position.
        pdec (Longitude): Losmandy physical declination encoder position.

    Returns:
        UnitSphericalRepresentation where longitude angle is the hour angle and the latitude angle
            is the declination in the mount reference frame.
    """
    if pdec < 180*u.deg:  # east of mount meridian
        mount_lon = (90*u.deg - pra) % (360*u.deg)
        mount_lat = Latitude(pdec - 90*u.deg)
    else:  # west of mount meridian
        mount_lon = (270*u.deg - pra) % (360*u.deg)
        mount_lat = Latitude(270*u.deg - pdec)

    return UnitSphericalRepresentation(mount_lon, mount_lat)


def residual(observation, model_params, location):
    """Compute the residual (error) between observed and modeled positions

    Args:
        observation: A Pandas Series containing a single observation.
        model_params (ModelParameters): Set of mount model parameters.
        location: An EarthLocation object.

    Returns:
        A Pandas Series containing a separation angle and position angle.
    """
    pra = Longitude(observation.mount_pra*u.deg)
    pdec = Longitude(observation.mount_pdec*u.deg)
    sc_mount = mount_to_world(
        losmandy_to_mount(pra, pdec),
        Time(observation.unix_timestamp, format='unix', location=location),
        model_params
    )
    sc_cam = SkyCoord(observation.solution_ra*u.deg, observation.solution_dec*u.deg, frame='icrs')

    return pd.Series([sc_mount.separation(sc_cam), sc_mount.position_angle(sc_cam)],
                     index=['separation', 'position_angle'])


def residuals(param_array, observations, location):
    """Generate series of residuals for a set of observations and model parameters.

    This is intended for use as the callback function passed to scipy.optimize.least_squares.

    Args:
        param_array (ndarray): Set of model parameters.
        observations (dataframe): Data from observations.
        location (EarthLocation): Observer location.

    Returns:
        A Pandas Series containing the magnitudes of the residuals in degrees.
    """
    res = observations.apply(
        residual,
        axis='columns',
        reduce=False,
        args=(ModelParameters.from_ndarray(param_array), location)
    ).separation
    return res.apply(lambda res_angle: res_angle.deg)


def plot_residuals(model_params, observations, location):
    """Plot the residuals on a polar plot.

    Args:
        model_params (ModelParameters): Set of model parameters.
        observations (dataframe): Data from observations.
        location (EarthLocation): Observer location.
    """
    res = observations.apply(
        residual,
        axis='columns',
        reduce=False,
        args=(model_params, location)
    )
    position_angles = res.position_angle.apply(lambda x: x.rad)
    separations = res.separation.apply(lambda x: x.arcmin)
    plt.polar(position_angles, separations, 'k.', label='residuals')
    plt.polar(np.linspace(0, 2*np.pi, 100), 90*np.ones(100), 'r', label='camera FOV')
    plt.title('Model Residuals (magnitude in arcminutes)')
    plt.legend()


class NoSolutionException(Exception):
    """Raised when optimization algorithm to solve for mount model parameters fails."""


def solve_model(observations, location):
    """Solves for mount model parameters using a set of observations and location.

    Finds a least-squares solution to the mount model parameters. The solution can then be used
    with the world_to_mount and mount_to_world functions in this module to convert between mount
    reference frame and celestial equatorial frame.

    Args:
        observations (dataframe): A set of observations where each contains a timestamp, mount
            encoder positions, and the corresponding celestial coordinates.
        location (EarthLocation): Location from which the observations were made.

    Returns:
        A ModelParameters object containing the solution.

    Raises:
        NoSolutionException if a solution could not be found.
    """

    # best starting guess for parameters
    init_values = ModelParameters(
        lon_axis_offset=Angle(0*u.deg),
        lat_axis_offset=Angle(0*u.deg),
        pole_rot_axis_lon=Angle(0*u.deg),
        pole_rot_angle=Angle(0*u.deg),
    )

    # lower bound on allowable values for each model parameter
    min_values = ModelParameters(
        lon_axis_offset=Angle(-180*u.deg),
        lat_axis_offset=Angle(-180*u.deg),
        pole_rot_axis_lon=Angle(-180*u.deg),
        pole_rot_angle=Angle(-180*u.deg),
    )

    # upper bound on allowable values for each model parameter
    max_values = ModelParameters(
        lon_axis_offset=Angle(180*u.deg),
        lat_axis_offset=Angle(180*u.deg),
        pole_rot_axis_lon=Angle(180*u.deg),
        pole_rot_angle=Angle(180*u.deg),
    )

    result = scipy.optimize.least_squares(
        residuals,
        init_values.to_ndarray(),
        bounds=(
            min_values.to_ndarray(),
            max_values.to_ndarray(),
        ),
        args=(observations, location),
    )

    if not result.success:
        raise NoSolutionException(result.message)

    return ModelParameters.from_ndarray(result.x), result
