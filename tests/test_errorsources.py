#!/usr/bin/env python3

"""Unit tests for track.errorsources"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude
from track.errorsources import separation, ErrorSource


class TestSeparation(unittest.TestCase):
    """Collection of unit tests for the track.model.MountModel class"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_separation(self):
        """Check that haversine-based separation function gives similar results as Astropy"""

        for _ in range(20):

            sc1 = SkyCoord(
                np.random.uniform(0.0, 360.0)*u.deg,
                np.random.uniform(-90.0, 90.0)*u.deg,
                frame='altaz',
            )
            sc2 = SkyCoord(
                np.random.uniform(0.0, 360.0)*u.deg,
                np.random.uniform(-90.0, 90.0)*u.deg,
                frame='altaz',
            )

            sep_astropy = sc1.separation(sc2)
            sep_track = separation(sc1, sc2)

            assert_allclose(sep_astropy.deg, sep_track.deg)

class TestSmallestAllowedError(unittest.TestCase):

    def test_positive_error(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(1*u.deg),
            target_enc_position=Longitude(2*u.deg)
        )
        assert_allclose(error.deg, 1.0)

    def test_negative_error(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(2*u.deg),
            target_enc_position=Longitude(1*u.deg)
        )
        assert_allclose(error.deg, -1.0)

    def test_wrapping(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(359*u.deg),
            target_enc_position=Longitude(0*u.deg)
        )
        assert_allclose(error.deg, 1.0)

    def test_target_closer_than_limit(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(0*u.deg),
            target_enc_position=Longitude(1*u.deg),
            no_cross_position=Longitude(30*u.deg),
        )
        assert_allclose(error.deg, 1.0)

    def test_target_opposite_dir_from_limit(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(0*u.deg),
            target_enc_position=Longitude(60*u.deg),
            no_cross_position=Longitude(-30*u.deg),
        )
        assert_allclose(error.deg, 60.0)

    def test_long_way_around(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude(-1*u.deg),
            target_enc_position=Longitude(1*u.deg),
            no_cross_position=Longitude(0*u.deg),
        )
        assert_allclose(error.deg, -358.0)

    def test_vector(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude([1, 2, 359], unit='deg'),
            target_enc_position=Longitude([2, 1, 0], unit='deg'),
        )
        assert_allclose(error.deg, np.array([1, -1, 1]))

    def test_vector_with_limit(self):
        error = ErrorSource._smallest_allowed_error(
            mount_enc_position=Longitude([1, 10, 10, 10], unit='deg'),
            target_enc_position=Longitude([2, 11, 90, -10], unit='deg'),
            no_cross_position=Longitude(0*u.deg)
        )
        assert_allclose(error.deg, np.array([1, 1, 80, 340]))

if __name__ == "__main__":
    unittest.main()
