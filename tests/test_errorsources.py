#!/usr/bin/env python3

"""Unit tests for track.errorsources"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from track.errorsources import separation


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

if __name__ == "__main__":
    unittest.main()
