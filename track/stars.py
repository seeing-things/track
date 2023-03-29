#!/usr/bin/env python3

"""
Identify bright stars near zenith.
"""

from __future__ import annotations
from importlib.resources import files
from astropy.time import Time
from configargparse import Namespace
import numpy as np
import pandas as pd
from skyfield.api import Star
from skyfield.positionlib import Barycentric
from track import gps_client, logs
from track.config import ArgParser
from track.skyfield_utils import astropy_to_skyfield_observer, hipparcos_df


class NoStarFoundException(Exception):
    """Raised when no star can be found meeting the criteria."""


def lookup_star_names(hip: int) -> tuple[str]:
    """Look up a star's common name given its Hipparcos ID (HIP)

    Args:
        hip: Hipparcos ID of star to look up.

    Returns:
        Tuple containing either one or two strings:
        - Common name of the star if found, and
        - The string 'HIP <N>' Where <N> is the HIP number.
    """
    df = pd.read_csv(files('track') / 'star_names_and_hips.csv', names=('name', 'hip'), header=0)
    names = []
    try:
        names.append(df[df.hip == hip].name.item())
    except ValueError:
        pass
    names.append(f'HIP {hip}')
    return tuple(names)


def lookup_star_magnitude(hip: int) -> float:
    """Lookup a star's apparent magnitude given its Hipparcos ID (HIP)"""
    return hipparcos_df[hipparcos_df.index == hip].magnitude.item()


def make_star_with_names(star_df: pd.Series) -> Star:
    """Transform a Hipparcos dataframe row to a Skyfield Star object with `names` populated.

    This is a wrapper around `Star.from_dataframe()` which also populates the `names` attribute of
    the `Star` object.

    Args:
        star_df: A row in the Hipparcos dataframe.

    Returns:
        A `Star` object with `names` populated.
    """
    star = Star.from_dataframe(star_df)
    star.names = lookup_star_names(star_df.name)
    return star


def find_high_bright_stars(
    observer: Barycentric,
    min_altitude: float,
    max_magnitude: float,
) -> pd.DataFrame:
    """Find a set of high, bright stars.

    Args:
        observer: Represents the position and time of the observer.
        min_altitude: Minimum allowed altitude in degrees.
        max_magnitude: Maximum allowed apparent magnitude.

    Returns:
        A dataframe containing all stars that meet the criteria.

    Raises:
        NoStarFoundException if no stars meet the criteria.
    """
    bright_stars_df = hipparcos_df[hipparcos_df['magnitude'] <= max_magnitude].copy()
    if bright_stars_df.empty:
        raise NoStarFoundException(f'No stars have magnitude <= {max_magnitude}')
    alt, _, _ = observer.observe(Star.from_dataframe(bright_stars_df)).apparent().altaz()
    # Adding a new column to the dataframe with the *current* altitude of each star
    bright_stars_df['alt_degrees'] = alt.degrees
    high_bright_stars_df = bright_stars_df[alt.degrees > min_altitude]
    if high_bright_stars_df.empty:
        raise NoStarFoundException(
            f'No stars with magnitude <= {max_magnitude} have altitude >= {min_altitude} degrees.'
        )
    return high_bright_stars_df


def find_highest_bright_star(
    observer: Barycentric,
    min_altitude: float = 60.0,
    max_magnitude: float = 2.5,
) -> Star:
    """Find the highest bright and high star.

    Args:
        observer: Represents the position and time of the observer.
        min_altitude: Minimum allowed altitude in degrees.
        max_magnitude: Maximum allowed apparent magnitude.

    Returns:
        The highest star which meets the magnitude and altitude constraints.

    Raises:
        NoStarFoundException if no stars meet the criteria.
    """
    high_bright_stars_df = find_high_bright_stars(observer, min_altitude, max_magnitude)
    highest_bright_star = high_bright_stars_df.iloc[np.argmax(high_bright_stars_df.alt_degrees)]
    return make_star_with_names(highest_bright_star)


def find_brightest_high_star(
    observer: Barycentric,
    min_altitude: float = 60.0,
    max_magnitude: float = 4.0,
) -> Star:
    """Find the brightest bright and high star.

    Args:
        observer: Represents the position and time of the observer.
        min_altitude: Minimum allowed altitude in degrees.
        max_magnitude: Maximum allowed apparent magnitude.

    Returns:
        The brightest star which meets the magnitude and altitude constraints.

    Raises:
        NoStarFoundException if no stars meet the criteria.
    """
    high_bright_stars_df = find_high_bright_stars(observer, min_altitude, max_magnitude)
    brightest_high_star = high_bright_stars_df.iloc[np.argmin(high_bright_stars_df.magnitude)]
    return make_star_with_names(brightest_high_star)


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments relevant to finding a star.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    star_group = parser.add_argument_group(
        title='Auto Star Picker Options',
        description='Criteria for automatic selection of a bright, high elevation star',
    )
    star_group.add_argument(
        '--star-min-altitude',
        help='minimum altitude of auto-selected star in degrees',
        default=60.0,
        type=float,
    )
    star_group.add_argument(
        '--star-max-magnitude',
        help='maximum apparent magnitude of auto-selected star',
        default=4.0,
        type=float,
    )


def make_star_from_args(args: Namespace, observer: Barycentric) -> Star:
    """Make a Skyfield `Star` object meeting the criteria given by the program args.

    Args:
        args: Set of program arguments.
        observer: Represents the location of the observer and the time.

    Returns:
        The brightest star meeting the specified magnitude and altitude criteria.
    """
    return find_brightest_high_star(
        observer=observer,
        min_altitude=args.star_min_altitude,
        max_magnitude=args.star_max_magnitude,
    )


def main():
    """See module docstring."""

    parser = ArgParser()
    gps_client.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    add_program_arguments(parser)
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'stars')

    # Get location of observer from arguments or from GPS
    location = gps_client.make_location_from_args(args)
    observer = astropy_to_skyfield_observer(location, Time.now())

    # Approach 1: Find star nearest zenith with magnitude below a threshold
    highest_bright_star = find_highest_bright_star(
        observer, args.star_min_altitude, args.star_max_magnitude
    )
    highest_bright_star_alt, _, _ = observer.observe(highest_bright_star).apparent().altaz()
    print(
        f'Highest star below magnitude {args.star_max_magnitude:.2f} is '
        f'{highest_bright_star.names[0]} '
        f'with altitude {highest_bright_star_alt.degrees:.2f} deg.'
    )

    # Approach 2: Find the brightest star with altitude greater than a threshold
    brightest_high_star = find_brightest_high_star(
        observer, args.star_min_altitude, args.star_max_magnitude
    )
    brightest_high_star_alt, _, _ = observer.observe(brightest_high_star).apparent().altaz()
    print(
        f'Brightest star above {args.star_min_altitude:.0f} deg altitude is '
        f'{brightest_high_star.names[0]} '
        f'with altitude {brightest_high_star_alt.degrees:.2f} deg.'
    )


if __name__ == "__main__":
    main()
