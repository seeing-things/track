#!/usr/bin/env python3

"""Unit tests for track.model"""

import unittest
from numpy.testing import assert_allclose
from astropy import units as u
from track.mounts import MeridianSide, MountEncoderPositions
from track.model import MountModel, ModelParamSet, ModelParameters


class TestMountModel(unittest.TestCase):
    """Collection of unit tests for the track.model.MountModel class"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_encoder_to_spherical(self):

        params = ModelParameters(
            axis_0_offset=0*u.deg,
            axis_1_offset=0*u.deg,
            pole_rot_axis_az=0*u.deg,  # does not affect the method under test
            pole_rot_angle=0*u.deg,    # does not affect the method under test
        )
        param_set = ModelParamSet(params, None, None, None)
        model = MountModel(param_set)

        with self.subTest(i=0):
            # Encoder initial values on cold start of a Losmandy G11 mount. In the typical
            # "counterweight down" startup position (corresponding to encoder offsets of 0), the
            # optical tube is pointed precisely in the direction of the mount's physical pole.
            encoders = MountEncoderPositions(180*u.deg, 180*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            # the longitude value and meridian_side are both "don't care"/undefined since this is
            # at a pole so intentionally not applying any assertions to those values
            assert_allclose(spherical_coord.lat.deg, 90.0)


        # Tests with Axis 1 90 degrees east of mount meridian

        with self.subTest(i=1):
            # Axis 0 still in startup position
            encoders = MountEncoderPositions(180*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 90.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=2):
            # Axis 0 is rotated 90 degrees counter-clockwise from startup position if viewed from
            # behind/below the OTA
            encoders = MountEncoderPositions(90*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 180.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=3):
            # Axis 0 is rotated 180 degrees from startup position. (This would normally be an
            # impossible position for a German equatorial mount)
            encoders = MountEncoderPositions(0*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 270.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)

        with self.subTest(i=4):
            # Axis 0 is rotated 90 degrees clockwise from startup position if viewed from behind/
            # below the OTA
            encoders = MountEncoderPositions(270*u.deg, 90*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 0.0)
            self.assertEqual(meridian_side, MeridianSide.EAST)


        # Tests with Axis 1 90 degrees west of mount meridian

        with self.subTest(i=5):
            # Axis 0 still in startup position
            encoders = MountEncoderPositions(180*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 270.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=6):
            # Axis 0 is rotated 90 degrees counter-clockwise from startup position if viewed from
            # behind/below the OTA
            encoders = MountEncoderPositions(90*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 0.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=7):
            # Axis 0 is rotated 180 degrees from startup position. (This would normally be an
            # impossible position for a German equatorial mount)
            encoders = MountEncoderPositions(0*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 90.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)

        with self.subTest(i=8):
            # Axis 0 is rotated 90 degrees clockwise from startup position if viewed from behind/
            # below the OTA
            encoders = MountEncoderPositions(270*u.deg, 270*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            assert_allclose(spherical_coord.lat.deg, 0.0)
            assert_allclose(spherical_coord.lon.deg, 180.0)
            self.assertEqual(meridian_side, MeridianSide.WEST)


        with self.subTest(i=9):
            # Axis 1 is rotated 180 degrees from startup position
            encoders = MountEncoderPositions(180*u.deg, 0*u.deg)
            spherical_coord, meridian_side = model.encoder_to_spherical(encoders)
            # As with the startup position, this is at a pole in the mount's spherical coordinate
            # system so the longitude coordinate and meridian side are undefined/don't care
            assert_allclose(spherical_coord.lat.deg, -90.0)


if __name__ == "__main__":
    unittest.main()
