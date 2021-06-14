#!/usr/bin/env python3

"""Unit tests for track.model"""

import random
import unittest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle, UnitSphericalRepresentation, SkyCoord, Longitude
from track.mounts import MeridianSide, MountEncoderPositions
from track.model import MountModel, ModelParamSet, ModelParameters, apply_guide_cam_alignment_error


def assertSphericalClose(us1, us2):
    """Check that two UnitSphericalRepresentation objects are nearly equal"""
    assert_allclose(us1.lon.deg, us2.lon.deg)
    assert_allclose(us1.lat.deg, us2.lat.deg)


class TestModelParamSet(unittest.TestCase):
    """Collection of unit tests for `track.model.ModelParamSet` and associated functions"""

    def test_apply_guide_cam_alignment_error(self):
        """Apply, then remove a guide cam alignment error and check that initial and final match"""

        for _ in range(20):
            original_param_set = ModelParamSet(ModelParameters(
                    axis_0_offset=Angle(np.random.uniform(0.0, 360.0)*u.deg),
                    axis_1_offset=Angle(np.random.uniform(0.0, 360.0)*u.deg),
                    pole_rot_axis_az=Angle(np.random.uniform(0.0, 360.0)*u.deg),
                    pole_rot_angle=Angle(np.random.uniform(-180.0, 180.0)*u.deg),
                    camera_tilt=Angle(np.random.uniform(-10.0, 10.0)*u.deg),
                ),
                guide_cam_orientation=Longitude(np.random.uniform(0.0, 360.0)*u.deg),
                location=None,
                timestamp=None,
            )

            align_error = Angle(
                (np.random.uniform(-5.0, 5.0) + 1j*np.random.uniform(-5.0, 5.0)) * u.deg
            )

            transformed_param_set = apply_guide_cam_alignment_error(
                old_params=original_param_set,
                guide_cam_align_error=align_error
            )

            restored_param_set = apply_guide_cam_alignment_error(
                old_params=transformed_param_set,
                guide_cam_align_error=Angle(0*u.deg)
            )

            assert_allclose(restored_param_set.guide_cam_align_error.deg, 0.0)

            original_model_params = original_param_set.model_params
            restored_model_params = restored_param_set.model_params
            for orig, restored in zip(original_model_params, restored_model_params):
                assert_allclose(orig.deg, restored.deg)


class TestMountModel(unittest.TestCase):
    """Collection of unit tests for the track.model.MountModel class"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_camera_tilt(self):
        """Apply, then remove camera tilt and check that result is same as starting coordinate"""

        param_set = ModelParamSet(None, None, None, None)
        model = MountModel(param_set)

        for _ in range(20):

            params = ModelParameters(
                axis_0_offset=Angle(0*u.deg),  # does not affect the method under test
                axis_1_offset=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_axis_az=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_angle=Angle(0*u.deg),    # does not affect the method under test
                camera_tilt=Angle(np.random.uniform(-90.0, 90.0)*u.deg),
            )
            model.model_params = params

            coord = UnitSphericalRepresentation(
                np.random.uniform(0.0, 360.0)*u.deg,
                np.random.uniform(-90.0, 90.0)*u.deg
            )
            meridian_side = random.choice((MeridianSide.WEST, MeridianSide.EAST))
            tilted_coord = model.apply_camera_tilt(coord, meridian_side)
            final_coord = model.remove_camera_tilt(tilted_coord, meridian_side)
            assertSphericalClose(coord, final_coord)

    def test_camera_tilt_inverse(self):
        """Remove, then apply camera tilt for reachable positions and check result"""

        param_set = ModelParamSet(None, None, None, None)
        model = MountModel(param_set)

        for _ in range(20):

            tilt = Angle(np.random.uniform(-90.0, 90.0)*u.deg)

            params = ModelParameters(
                axis_0_offset=Angle(0*u.deg),  # does not affect the method under test
                axis_1_offset=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_axis_az=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_angle=Angle(0*u.deg),    # does not affect the method under test
                camera_tilt=tilt,
            )
            model.model_params = params

            coord = UnitSphericalRepresentation(
                np.random.uniform(0.0, 360.0)*u.deg,
                np.random.uniform(-90.0 + abs(tilt.deg), 90.0 - abs(tilt.deg))*u.deg
            )
            meridian_side = random.choice((MeridianSide.WEST, MeridianSide.EAST))
            untilted_coord = model.remove_camera_tilt(coord, meridian_side)
            final_coord = model.apply_camera_tilt(untilted_coord, meridian_side)
            assertSphericalClose(coord, final_coord)


    def test_camera_tilt_unreachable(self):
        """Remove, then apply camera tilt for unreachable positions and check separation"""

        param_set = ModelParamSet(None, None, None, None)
        model = MountModel(param_set)

        for _ in range(20):

            tilt = Angle(np.random.uniform(-90.0, 90.0)*u.deg)

            params = ModelParameters(
                axis_0_offset=Angle(0*u.deg),  # does not affect the method under test
                axis_1_offset=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_axis_az=Angle(0*u.deg),  # does not affect the method under test
                pole_rot_angle=Angle(0*u.deg),    # does not affect the method under test
                camera_tilt=tilt,
            )
            model.model_params = params

            coord = UnitSphericalRepresentation(
                np.random.uniform(0.0, 360.0)*u.deg,
                random.choice((-1, +1)) * np.random.uniform(90.0 - abs(tilt.deg), 90.0)*u.deg
            )
            meridian_side = random.choice((MeridianSide.WEST, MeridianSide.EAST))
            untilted_coord = model.remove_camera_tilt(coord, meridian_side)
            final_coord = model.apply_camera_tilt(untilted_coord, meridian_side)

            separation = SkyCoord(coord).separation(SkyCoord(final_coord))
            expected_separation = np.abs(tilt) - (90*u.deg - np.abs(coord.lat))
            assert_allclose(separation.deg, expected_separation.deg)


    def test_encoders_to_spherical(self):

        params = ModelParameters(
            axis_0_offset=Angle(0*u.deg),
            axis_1_offset=Angle(0*u.deg),
            pole_rot_axis_az=Angle(0*u.deg),  # does not affect the method under test
            pole_rot_angle=Angle(0*u.deg),    # does not affect the method under test
            camera_tilt=Angle(0*u.deg),       # does not affect the method under test
        )
        param_set = ModelParamSet(params, None, None, None)
        model = MountModel(param_set)

        with self.subTest(i=0):
            # Encoder initial values on cold start of a Losmandy G11 mount. In the typical
            # "counterweight down" startup position (corresponding to encoder offsets of 0), the
            # optical tube is pointed precisely in the direction of the mount's physical pole.
            encoders = MountEncoderPositions(180*u.deg, 180*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            # the longitude value and meridian_side are both "don't care"/undefined since this is
            # at a pole so intentionally not applying any assertions to those values
            assert_allclose(spherical_coord.lat.deg, 90.0)


        # Tests with Axis 1 90 degrees east of mount meridian

        with self.subTest(i=1):
            # Axis 0 still in startup position
            encoders = MountEncoderPositions(180*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 90.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=2):
            # Axis 0 is rotated 90 degrees counter-clockwise from startup position if viewed from
            # behind/below the OTA
            encoders = MountEncoderPositions(90*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 180.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=3):
            # Axis 0 is rotated 180 degrees from startup position. (This would normally be an
            # impossible position for a German equatorial mount)
            encoders = MountEncoderPositions(0*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 270.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=4):
            # Axis 0 is rotated 90 degrees clockwise from startup position if viewed from behind/
            # below the OTA
            encoders = MountEncoderPositions(270*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 0.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)


        # Tests with Axis 1 90 degrees west of mount meridian

        with self.subTest(i=5):
            # Axis 0 still in startup position
            encoders = MountEncoderPositions(180*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 270.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=6):
            # Axis 0 is rotated 90 degrees counter-clockwise from startup position if viewed from
            # behind/below the OTA
            encoders = MountEncoderPositions(90*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 0.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=7):
            # Axis 0 is rotated 180 degrees from startup position. (This would normally be an
            # impossible position for a German equatorial mount)
            encoders = MountEncoderPositions(0*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 90.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=8):
            # Axis 0 is rotated 90 degrees clockwise from startup position if viewed from behind/
            # below the OTA
            encoders = MountEncoderPositions(270*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 180.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)


        with self.subTest(i=9):
            # Axis 1 is rotated 180 degrees from startup position
            encoders = MountEncoderPositions(180*u.deg, 0*u.deg)
            spherical_coord, meridian_side = model.encoders_to_spherical(encoders)
            # As with the startup position, this is at a pole in the mount's spherical coordinate
            # system so the longitude coordinate and meridian side are undefined/don't care
            assert_allclose(spherical_coord.lat.deg, -90.0)


if __name__ == "__main__":
    unittest.main()
