"""Mount modeling and transformations between sky and mount reference frames"""

import os
import pickle
import time
from typing import NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
from astropy.utils import iers
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Longitude, Angle, EarthLocation, AltAz
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates.representation import UnitSphericalRepresentation, CartesianRepresentation
from track.config import CONFIG_PATH
from track.mounts import MeridianSide, MountEncoderPositions


DEFAULT_MODEL_FILENAME = os.path.join(CONFIG_PATH, 'model_params.pickle')


# Disable IERS data age checking in calls to methods that depend on this data so that the code will
# still work when offline. Even when online this is helpful as otherwise the first call to a method
# in Astropy that uses IERS data will block and wait for the update mechanism to download the
# latest data, and such delays could be problematic. Note however that it is necessary to
# download this at least once using `iers.IERS_Auto().open()`, otherwise the affected methods will
# still fail. (Previously the aforementioned method call was invoked here such that it runs when
# this module is imported such that the data would at least be downloaded when the machine has
# internet access, but this causes the import to take a second or two to complete which is
# annoying so it was removed.) For exacting pointing accuracy this data should be updated
# regularly. See https://docs.astropy.org/en/stable/utils/iers.html for additional details.
iers.conf.auto_max_age = None  # Disable strict stale cache check


class ModelParameters(NamedTuple):
    """Set of parameters for the mount model.

    When paired with the equations in the topocentric_to_encoders and encoders_to_topocentric
    methods of the MountModel class these parameters define a unique transformation between mount
    encoder positions and coordinates in the topocentric coordinate system (azimuth and altitude).

    Attributes:
        axis_0_offset: Encoder zero-point offset for the longitude mount axis. For equatorial
            mounts this is the right ascension axis. For altazimuth mounts this is the azimuth
            axis.
        axis_1_offset: Encoder zero-point offset for the latitude mount axis. For equatorial
            mounts this is the declination axis. For altazimuth mounts this is the altitude axis.
        pole_rot_axis_az: The azimuthal angle of the axis of rotation used to transform from a
            spherical coordinate system using the mount physical axes to the topocentric coordinate
            system.
        pole_rot_angle: The angular separation between the instrument pole and local zenith. For
            altazimuth mounts this will be close to zero. For equatorial mounts oriented with the
            instrument pole aligned with the celestial pole this will be approximately 90 degrees
            minus the latitude of the observer.
        camera_tilt: Tilt of the camera away from the plane that is perpendicular to axis 1.
    """
    axis_0_offset: Angle
    axis_1_offset: Angle
    pole_rot_axis_az: Angle
    pole_rot_angle: Angle
    camera_tilt: Angle

    @staticmethod
    def from_ndarray(param_array: np.ndarray) -> "ModelParameters":
        """Factory method to generate an instance of this class with values given in an ndarray.

        Args:
            param_array (ndarray): An array of parameter values. This format is required when
                interfacing with the scipy least_squares method.
        """
        return ModelParameters(
            axis_0_offset=Angle(param_array[0]*u.deg),
            axis_1_offset=Angle(param_array[1]*u.deg),
            pole_rot_axis_az=Angle(param_array[2]*u.deg),
            pole_rot_angle=Angle(param_array[3]*u.deg),
            camera_tilt=Angle(param_array[4]*u.deg),
        )

    def to_ndarray(self) -> np.ndarray:
        """Return an ndarray containing the model parameters.

        The output format is suitable for use with scipy least_squares.

        Returns:
            An ndarray object containing the parameter values.
        """
        return np.array([
            self.axis_0_offset.deg,
            self.axis_1_offset.deg,
            self.pole_rot_axis_az.deg,
            self.pole_rot_angle.deg,
            self.camera_tilt.deg,
        ])


class ModelParamSet(NamedTuple):
    """Collection of mount model parameters and the location and time where they were generated.

    The purpose of this class is to pair an instance of ModelParameters with the location where
    they were generated and an approximate time of generation. An instance of this object can be
    pickled and stored on disk for later usage. The location is important because the model
    parameters are only valid for that location. The timestamp is included to allow for checks on
    the freshness of the parameters--for example, if they are more than a few hours old they may
    no longer be trustworthy unless the mount is a permanent installation.

    Attributes:
        model_params: Instance of ModelParameters as defined in this module.
        guide_cam_orientation: The orientation of the guidescope camera with respect to the mount-
            relative spherical coordinate system (as returned by, e.g.,
            `MountModel.encoder_to_spherical()`). Specifically, this gives the position angle of
            the positive x-axis in the camera frame, where "position angle" is defined using the
            east-of-north convention.
        location: Location of the mount for which model_params is applicable. Not used in any
            MountModel calculations.
        timestamp: Unix timestamp giving the approximate time that this set of model parameters
            was generated. Used only to check if the model parameters are stale; not used in any
            MountModel calculations.
    """
    model_params: ModelParameters
    guide_cam_orientation: Longitude
    location: EarthLocation
    timestamp: Optional[float]


def tip_axis(
        coord: UnitSphericalRepresentation,
        axis_lon: Angle,
        rot_angle: Angle
    ) -> UnitSphericalRepresentation:
    """Perform a rotation about an axis perpendicular to the Z-axis.

    The purpose of this rotation is to move the pole of the coordinate system from one place to
    another. For example, transforming from a coordinate system where the pole is aligned with the
    physical pole of a mount to a topocentric coordinate system where the pole is aligned with
    zenith.

    Note that this is a true rotation, and the same rotation is applied to all coordinates in the
    originating coordinate system equally. It is not equivalent to using SkyCoord
    directional_offset_by() with a fixed position angle and separation since the direction and
    magnitude of the offset depend on the value of coord.

    Args:
        coord: Coordinate to be transformed.
        axis_lon: Longitude angle of the axis of rotation.
        rot_angle: Angle of rotation.

    Returns:
        Coordinate after transformation.
    """
    rot = rotation_matrix(
        rot_angle,
        axis=SkyCoord(axis_lon, 0*u.deg).represent_as('cartesian').xyz
    )
    coord_cart = coord.represent_as(CartesianRepresentation)
    coord_rot_cart = coord_cart.transform(rot)
    return coord_rot_cart.represent_as(UnitSphericalRepresentation)


class MountModel:
    """A math model of a telescope mount.

    This class provides transformations between mount encoder position readings and coordinates in
    topocentric (AzAlt) frame.

    Attributes:
        model_params (ModelParameters): The set of parameters to be used in the transformations.
        guide_cam_orientation (Longitude): The orientation of the guidescope camera.
        location (EarthLocation): The observer location for which the mount model is correct.
    """


    def __init__(self, model_param_set: ModelParamSet):
        """Construct an instance of MountModel

        Args:
            model_param_set: Set of model parameters to use in calculations.
        """
        self.model_params = model_param_set.model_params
        self.guide_cam_orientation = model_param_set.guide_cam_orientation
        self.location = model_param_set.location


    def apply_camera_tilt(
            self,
            coord: UnitSphericalRepresentation,
            meridian_side: MeridianSide,
        ) -> UnitSphericalRepresentation:
        """Applies the effect of camera tilt.

        This method takes a coordinate in the mount-relative spherical coordinate system
        corresponding to where the camera would be pointed if it had no tilt and transforms it to
        where the camera is actually pointed due to tilt. In the context of this method, tilt is
        defined as rotation of the camera away from the plane that is perpendicular to axis 1.

        The transformation is a directional offset in the direction of a reference position that
        is on the equator of the spherical coordiante system and having a longitude that is 90
        degrees offset from the input coordinate. For a given tilt and meridian side, all input
        coordinates having the same longitude will be moved towards this reference position. In
        other words, points that started out along the arc of a great circle will be shifted
        such that if they are connected to the center of the sphere they would form a cone centered
        on the reference position.

        Args:
            coord: Coordinate in the mount-relative spherical coordinate system to be transformed,
                representing the direction the camera would be pointed if there were no tilt.
            meridian_side: Side of mount meridian.

        Returns:
            Coordinate corresponding to the position the camera is actually pointed.
        """
        sc = SkyCoord(coord)
        if abs(coord.lat.deg) < 90.0:
            lon_offset = 90*u.deg if meridian_side == MeridianSide.WEST else -90*u.deg
        else:
            lon_offset = 0*u.deg
        reference_coord = SkyCoord(coord.lon + lon_offset, 0*u.deg)
        return sc.directional_offset_by(
            position_angle=sc.position_angle(reference_coord),
            separation=self.model_params.camera_tilt
        ).represent_as(UnitSphericalRepresentation)


    def remove_camera_tilt(
            self,
            coord: UnitSphericalRepresentation,
            meridian_side: MeridianSide,
        ) -> UnitSphericalRepresentation:
        """Removes the effect of camera tilt.

        This method takes a coordinate in the mount-relative spherical coordinate system and
        transforms it to where the camera would be pointed in that coordinate system if the
        camera had no tilt. This transformation is one of the steps towards determining what
        encoder positions correspond to a particular camera position.

        Note that camera tilt results in regions near the poles of the coordinate system that are
        unreachable by the mount. Specifically, input coordinates with latitudes that are within
        the camera tilt angle of either pole are unreachable. When such a coordinate is encountered
        this method will return the pole location which is the nearest reachable position.

        The formula for the longitude of the reference point was found by noting that the reference
        point has latitude 0 and its separation from the input coordinate is equal to 90 degrees
        minus the camera tilt angle. With this information and the equation for the great circle
        distance between two points on a sphere, it is possible to solve for the difference in
        longitude between the input coordinate and the reference point.

        Args:
            coord: Coordinate in the mount-relative spherical coordinate system to be transformed,
                representing the direction the camera is pointed.
            meridian_side: Side of mount meridian.

        Returns:
            Coordinate corresponding to the position the camera would be pointed if it had no tilt,
            or the nearest position that is reachable.
        """
        tilt = self.model_params.camera_tilt

        # Positions meeting these criteria are not reachable; return nearest reachable position
        if coord.lat >= 90*u.deg - np.abs(tilt):
            return UnitSphericalRepresentation(coord.lon, 90*u.deg)
        if coord.lat <= -(90*u.deg - np.abs(tilt)):
            return UnitSphericalRepresentation(coord.lon, -90*u.deg)

        # difference in longitude of coord and reference position longitude
        lon_diff = np.arccos(np.cos(90*u.deg - tilt) / np.cos(coord.lat))

        # the tilt is in the direction of this reference position
        reference_coord = SkyCoord(
            coord.lon + lon_diff if meridian_side == MeridianSide.WEST else coord.lon - lon_diff,
            0*u.deg,
        )
        sc = SkyCoord(coord)
        return sc.directional_offset_by(
            position_angle=sc.position_angle(reference_coord),
            separation=-self.model_params.camera_tilt
        ).represent_as(UnitSphericalRepresentation)


    def encoder_to_spherical(
            self,
            encoder_positions: MountEncoderPositions
        ) -> Tuple[UnitSphericalRepresentation, MeridianSide]:
        """Convert from mount encoder positions to mount-relative spherical coordinates.

        The mount-relative spherical coordinate system is a defined such that the positive Z-axis,
        corresponding to a latitude angle of +90 degrees, is aligned with the physical pole of the
        mount, wherever it may be.

        Special cases: For an equatorial mount with the pole tipped up to be aligned with zenith
        the spherical coordinates returned by this function can be interpreted as topocentric
        azimuth and altitude. Alternatively, they can be interpreted as local hour angle and
        declination if used with an equatorial mount where the pole of the mount is perfectly
        aligned with the celestial pole after applying a 180-degree shift to the longitude
        coordinate.

        The details of the transformation applied here follow the conventions used by the Losmandy
        G11 mount's "physical" encoder position "pra" and "pdec". In particular, the default
        starting values of the encoders in the "counterweight down" startup position and the
        direction of axis motion corresponding to increasing encoder reading. This should work with
        other mounts as long as the "handedness" of the encoders is the same. The encoder zero
        point offsets in the mount model should take care of any difference in the startup
        positions.

        Args:
            encoder_positions: Set of mount encoder positions to be converted.

        Returns:
            Tuple where first element is the spherical coordinate, where longitude corresponds to
            mount axis 0 and latitude corresponds to mount axis 1, and the second element is the
            meridian side this position lies on.
        """

        # apply encoder offsets
        encoder_positions = MountEncoderPositions(
            Longitude(encoder_positions[0] - self.model_params.axis_0_offset),
            Longitude(encoder_positions[1] - self.model_params.axis_1_offset),
        )

        # TODO: This transformation is only correct if the mount axes are exactly orthogonal. This
        # should be replaced with a more general transformation that can handle non-orthogonal axes.
        if encoder_positions[1] < 180*u.deg:
            meridian_side = MeridianSide.EAST
            spherical_coord = UnitSphericalRepresentation(
                lon=(270*u.deg - encoder_positions[0]),
                lat=(encoder_positions[1] - 90*u.deg)
            )
        else:
            meridian_side = MeridianSide.WEST
            spherical_coord = UnitSphericalRepresentation(
                lon=(90*u.deg - encoder_positions[0]),
                lat=(270*u.deg - encoder_positions[1])
            )

        return self.apply_camera_tilt(spherical_coord, meridian_side), meridian_side


    def spherical_to_encoder(
            self,
            coord: UnitSphericalRepresentation,
            meridian_side=MeridianSide.EAST,
        ) -> MountEncoderPositions:
        """Convert from mount-relative spherical coordinates to mount encoder positions

        See docstring of encoder_to_spherical, which is the inverse of this method.

        Not all positions in the spherical coordinate system are necessarily reachable by the
        mount. For example, camera tilt or mount axes that are not perfectly orthogonal may create
        regions near the mount poles that are unreachable. In such cases this method will return
        the encoder positions that correspond to the nearest reachable position.

        Args:
            coord: Coordinate in the mount frame.
            meridian_side: Desired side of mount-relative meridian. If the pole of the mount is not
                in the direction of the celestial pole this may not correspond to true east and
                west directions.

        Returns:
            Mount encoder positions corresponding to coord.
        """
        if not isinstance(meridian_side, MeridianSide):
            raise TypeError('meridian_side should be a MeridianSide!!')

        coord = self.remove_camera_tilt(coord, meridian_side)

        # TODO: This transformation is only correct if the mount axes are exactly orthogonal. This
        # should be replaced with a more general transformation that can handle non-orthogonal axes.
        if meridian_side == MeridianSide.EAST:
            encoder_0 = Longitude(270*u.deg - coord.lon)
            encoder_1 = Longitude(90*u.deg + coord.lat)
        else:
            encoder_0 = Longitude(90*u.deg - coord.lon)
            encoder_1 = Longitude(270*u.deg - coord.lat)

        # apply encoder offsets
        return MountEncoderPositions(
            Longitude(encoder_0 + self.model_params.axis_0_offset),
            Longitude(encoder_1 + self.model_params.axis_1_offset),
        )


    def encoders_to_topocentric(self, encoder_positions: MountEncoderPositions) -> SkyCoord:
        """Convert mount encoder positions to a topocentric coordinate.

        Args:
            encoder_positions (MountEncoderPositions): Set of mount encoder positions.

        Returns:
            A SkyCoord object with AltAz frame. The location and obstime attributes of this object
            will not be populated. These will need to be set in order to transform it to an
            inertial equatorial frame such as ICRS.
        """

        # convert encoder positions to mount-relative spherical coordinate system
        mount_coord, _ = self.encoder_to_spherical(encoder_positions)

        # transform pole of coordinate system from mount pole to local zenith
        return SkyCoord(tip_axis(
            mount_coord,
            self.model_params.pole_rot_axis_az,
            -self.model_params.pole_rot_angle
        ), frame=AltAz)


    def topocentric_to_encoders(
            self,
            sky_coord: SkyCoord,
            meridian_side: MeridianSide,
        ) -> MountEncoderPositions:
        """Convert coordinate for a position on the sky to mount encoder positions.

        Args:
            sky_coord: Coordinate in AltAz frame to be converted.
            meridian_side: Gives the desired side of the meridian to use (for equatorial mounts).

        Returns:
            Encoder positions corresponding to this sky coordinate and on the desired side of the
            meridian.

        Raises:
            TypeError if frame of sky_coord is not AltAz.
        """

        if not isinstance(sky_coord.frame, AltAz):
            raise TypeError('frame of sky_coord must be AltAz')

        # transform pole of coordinate system from local zenith to mount pole
        mount_coord = tip_axis(
            sky_coord,
            self.model_params.pole_rot_axis_az,
            self.model_params.pole_rot_angle
        )

        # transform from mount-relative spherical coordiante to encoder positions
        return self.spherical_to_encoder(mount_coord, meridian_side)


def ha_to_ra(hour_angle: Longitude, longitude: Longitude, t: Time) -> Longitude:
    """Converts hour angle to right ascension

    Args:
        hour_angle: The hour angle to be converted.
        longitude: The longitude of the observer.
        t: Time of observation.

    Returns:
        The right ascension angle.
    """
    return Longitude(t.sidereal_time('apparent', longitude=longitude) - hour_angle)


def ra_to_ha(ra_angle: Longitude, longitude: Longitude, t: Time) -> Longitude:
    """Converts right ascension to hour angle

    Args:
        ra_angle: The right ascension angle to be converted.
        longitude: The longitude of the observer.
        t: Time of observation.

    Returns:
        The hour angle.
    """
    # the operation is its own inverse
    return ha_to_ra(ra_angle, longitude, t)


def residual(
        observation: pd.Series,
        model_params: ModelParameters,
    ) -> pd.Series:
    """Compute the residual (error) between observed and modeled positions

    Args:
        observation: A single observation.
        model_params: Set of mount model parameters.

    Returns:
        A Pandas Series containing a separation angle and position angle.
    """
    encoder_positions = MountEncoderPositions(
        Longitude(observation.encoder_0*u.deg),
        Longitude(observation.encoder_1*u.deg),
    )
    mount_model = MountModel(ModelParamSet(model_params, None, None, None))
    sc_mount = mount_model.encoders_to_topocentric(encoder_positions)
    sc_cam = SkyCoord(observation.sky_az*u.deg, observation.sky_alt*u.deg, frame='altaz')

    return pd.Series([sc_mount.separation(sc_cam), sc_mount.position_angle(sc_cam)],
                     index=['separation', 'position_angle'])


def residuals(
        param_array: np.ndarray,
        observations: pd.DataFrame,
    ) -> pd.Series:
    """Generate series of residuals for a set of observations and model parameters.

    This is intended for use as the callback function passed to scipy.optimize.least_squares.

    Args:
        param_array: Set of model parameters.
        observations: Data from observations.

    Returns:
        A Pandas Series containing the magnitudes of the residuals in degrees.
    """
    res = observations.apply(
        residual,
        axis='columns',
        reduce=False,
        args=(ModelParameters.from_ndarray(param_array),)
    ).separation
    return res.apply(lambda res_angle: res_angle.deg)


def plot_residuals(
        model_params: ModelParameters,
        observations: pd.DataFrame,
    ) -> None:
    """Plot the residuals on a polar plot.

    Args:
        model_params: Set of model parameters.
        observations: Data from observations.
    """
    res = observations.apply(
        residual,
        axis='columns',
        reduce=False,
        args=(model_params,)
    )
    position_angles = res.position_angle.apply(lambda x: x.rad)
    separations = res.separation.apply(lambda x: x.arcmin)
    plt.polar(position_angles, separations, 'k.', label='residuals')
    plt.polar(np.linspace(0, 2*np.pi, 100), 90*np.ones(100), 'r', label='camera FOV')
    plt.title('Model Residuals (magnitude in arcminutes)')
    plt.legend()


class NoSolutionException(Exception):
    """Raised when optimization algorithm to solve for mount model parameters fails."""


def solve_model(observations: pd.DataFrame) -> Tuple[ModelParameters, OptimizeResult]:
    """Solves for mount model parameters using a set of observations.

    Finds a least-squares solution to the mount model parameters. The solution can then be used
    with the topocentric_to_encoders and encoders_to_topocentric functions in this module to
    convert between mount reference frame and celestial equatorial frame.

    Args:
        observations: A set of observations where each contains a timestamp, mount encoder
            positions, and the corresponding topocentric coordinates.

    Returns:
        The parameters for the solution.

    Raises:
        NoSolutionException if a solution could not be found.
    """

    # best starting guess for parameters
    init_values = ModelParameters(
        axis_0_offset=Angle(0*u.deg),
        axis_1_offset=Angle(0*u.deg),
        pole_rot_axis_az=Angle(0*u.deg),
        pole_rot_angle=Angle(0*u.deg),
        camera_tilt=Angle(0*u.deg),
    )

    # lower bound on allowable values for each model parameter
    min_values = ModelParameters(
        axis_0_offset=Angle(-180*u.deg),
        axis_1_offset=Angle(-180*u.deg),
        pole_rot_axis_az=Angle(-180*u.deg),
        pole_rot_angle=Angle(-180*u.deg),
        camera_tilt=Angle(-10*u.deg),
    )

    # upper bound on allowable values for each model parameter
    max_values = ModelParameters(
        axis_0_offset=Angle(180*u.deg),
        axis_1_offset=Angle(180*u.deg),
        pole_rot_axis_az=Angle(180*u.deg),
        pole_rot_angle=Angle(180*u.deg),
        camera_tilt=Angle(10*u.deg),
    )

    result = scipy.optimize.least_squares(
        residuals,
        init_values.to_ndarray(),
        bounds=(
            min_values.to_ndarray(),
            max_values.to_ndarray(),
        ),
        args=(observations,),
    )

    if not result.success:
        raise NoSolutionException(result.message)

    return ModelParameters.from_ndarray(result.x), result


class StaleParametersException(Exception):
    """Raised when model parameters are stale and potentially invalid"""


def save_default_param_set(model_param_set: ModelParamSet) -> None:
    """Saves the model parameter set to disk at the default location.

    This overwrites any model parameters already saved at this location.

    Args:
        model_param_set: Parameter set to save.
    """
    with open(DEFAULT_MODEL_FILENAME, 'wb') as f:
        pickle.dump(model_param_set, f, pickle.HIGHEST_PROTOCOL)


def load_default_param_set(max_age: Optional[float] = 12*3600) -> ModelParamSet:
    """Loads the model parameter set from disk at the default location.

    Args:
        max_age: Max allowed age of the model parameters in seconds. If the parameter set is older
            than this an exception is raised. If None no check is performed.

    Returns:
        Parameter set loaded from disk.

    Raises:
        StaleParametersException: When the timestamp of the ModelParamSet loaded from disk exceeds
            max_age.
    """
    with open(DEFAULT_MODEL_FILENAME, 'rb') as f:
        model_param_set = pickle.load(f)

    if max_age is not None:
        param_set_age = time.time() - model_param_set.timestamp
        if param_set_age > max_age:
            raise StaleParametersException(
                'model params are {:.1f} hours old'.format(param_set_age / 3600.0))

    return model_param_set


def load_default_model(
        max_age: Optional[float] = 12*3600
    ) -> MountModel:
    """Loads the model parameter set from disk and returns a MountModel instance.

    Args:
        max_age: Max allowed age of the model parameters in seconds. If the parameter set is older
            than this an exception is raised. If None no check is performed.

    Returns:
        A MountModel instantiated with the parameters loaded from disk.

    Raises:
        StaleParametersException: When the timestamp of the ModelParamSet loaded from disk exceeds
        max_age.
    """
    return MountModel(load_default_param_set(max_age))
