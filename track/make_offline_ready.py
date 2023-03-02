#!/usr/bin/env python3

"""Download data files required for offline use."""

import logging
import astropy.utils.iers
from track import logs
from track.config import ArgParser
from track.skyfield_utils import download_skyfield_data_files


logger = logging.getLogger(__name__)


def download_astropy_iers() -> None:
    """Download fresh Astropy IERS data.

    Runs the Astropy code that downloads fresh International Earth Rotation and Reference Systems
    (IERS) data, which is how Astropy learns about Earth's polar motion and rotation rate since
    this evolves in an unpredictable manner at the fine-grained level. Due to the unpredictable
    nature of this data, it will get stale, so for highest accuracy fresh IERS data could be
    downloaded from time to time. That said, fresh IERS data should be unnecessary to track
    satellites, which is the main focus of this package.

    See the following for more info:
    https://docs.astropy.org/en/stable/utils/iers.html
    https://docs.astropy.org/en/stable/api/astropy.utils.iers.IERS_Auto.html
    https://docs.astropy.org/en/stable/api/astropy.utils.iers.LeapSeconds.html
    """
    astropy.utils.iers.IERS_Auto().open()
    astropy.utils.iers.LeapSeconds.auto_open()


def main() -> None:
    """Downloads data files required by this package such that they are available offline.

    Certain dependencies of this package rely on data files that are downloaded from the internet.
    This function will download all of these to disk such that no subsequent downloads should be
    required. Once this has been called and all files are downloaded successfully, most features of
    this package should work without an internet connection.
    """
    parser = ArgParser()
    logs.add_program_arguments(parser)
    args = parser.parse_args()
    logs.setup_logging_from_args(args, 'make_offline_ready')

    logger.info('Downloading Astropy data files.')
    download_astropy_iers()
    logger.info('Downloading Skyfield data files.')
    download_skyfield_data_files()
    logger.info('All files required for offline use have been downloaded.')


if __name__ == "__main__":
    main()
