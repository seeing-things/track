"""Skyfield utilities"""


import logging
import os
from astropy.coordinates import EarthLocation
from astropy.time import Time
import skyfield.api
from skyfield.api import load, wgs84
from skyfield.data import hipparcos
from skyfield.positionlib import Barycentric
from track.config import DATA_PATH


logger = logging.getLogger(__name__)


# Data files required by Skyfield
DE421_FILENAME = os.path.join(DATA_PATH, 'de421.bsp')
HIPPARCOS_FILENAME = os.path.join(DATA_PATH, 'hipparcos.dat')


skyfield_loader = skyfield.api.Loader(DATA_PATH)
planets = skyfield_loader('de421.bsp')
timescale = skyfield_loader.timescale()

# Load the Hipparcos catalog which contains about 118,000 stars.
with load.open(hipparcos.URL, filename=HIPPARCOS_FILENAME) as f:
    hipparcos_df = hipparcos.load_dataframe(f)
# Some stars in the Hipparcos catalog don't have all fields defined
hipparcos_df = hipparcos_df[hipparcos_df['ra_degrees'].notnull()]


def download_skyfield_data_files() -> None:
    """Download data files required by Skyfield in this package."""
    skyfield.api.load.download('de421.bsp', filename=DE421_FILENAME)
    skyfield.api.load.download(skyfield.data.hipparcos.URL, filename=HIPPARCOS_FILENAME)


def astropy_to_skyfield_observer(location: EarthLocation, time: Time) -> Barycentric:
    """Convert Astropy location and time to a Skyfield object.

    Args:
        location: Location on Earth's surface.
        time: A specific time.

    Returns:
        A Skyfield object representing an observer's location and time.
    """
    geographic_position = wgs84.latlon(location.lat.deg, location.lon.deg, location.height.value)
    location_skyfield = planets['earth'] + geographic_position
    time_skyfield = timescale.from_astropy(time)
    return location_skyfield.at(time_skyfield)
