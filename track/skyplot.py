#!/usr/bin/env python3

"""Topocentric plots of the sky for planning and post-mortem debugging.

This file can generate plots of the sky showing information that is relevant to tracking of various
object but especially satellite passes. Information includes regions of the sky reachable by the
mount within axis limits, trajectory of satellite pass from TLE, and trajectory of the mount from a
completed (or even in-progress) pass.
"""

from typing import Tuple, Optional
from datetime import datetime, timedelta
from configargparse import ArgParser
import dateutil
import numpy as np
import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import ephem
from astropy import units as u
from astropy.coordinates import Longitude, UnitSphericalRepresentation, EarthLocation
import track
from track import gps_client, telem
from track.mounts import MountEncoderPositions, MeridianSide
from track.model import MountModel


def add_arrow(line: matplotlib.lines.Line2D, size: int = 15, color: str = None) -> None:
    """Add an arrow to a line.

    Args:
        line: Line2D object
        color: If None, line color is taken
    """
    if color is None:
        color = line.get_color()

    theta = line.get_xdata()
    radius = line.get_ydata()

    # place arrow near max altitude (min radius) point on the curve
    ind = np.argmin(radius)

    if radius.size < 2:
        # can't annotate a line with length 1
        return
    elif ind == 0:
        ind += 1

    line.axes.annotate('',
        xytext=(theta[ind-1], radius[ind-1]),
        xy=(theta[ind], radius[ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )


def plot_trajectory(
        ax: matplotlib.axes.Axes,
        az: np.ndarray,
        alt: np.ndarray,
        color: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
    """Plot a curve with an arrow indicating direction of motion."""
    line = ax.plot(np.radians(az), 90.0 - alt, color=color, label=label)[0]
    add_arrow(line)


def fill_to_horizon(
        ax: matplotlib.axes.Axes,
        az: np.ndarray,
        alt: np.ndarray,
        alpha: float = 1.0,
        color=None
    ) -> None:
    """Fill the region between a curve and the horizon."""
    az = az[alt >= 0]
    alt = alt[alt >= 0]
    ax.fill_between(
        np.radians(az),
        90.0 - alt,
        100*np.ones_like(az),
        alpha=alpha,
        color=color,
        linewidth=0,
    )


def fill_to_zenith(
        ax: matplotlib.axes.Axes,
        az: np.ndarray,
        alt: np.ndarray,
        alpha: float = 1.0,
        color: Optional[str] = None
    ) -> None:
    """Fill the region between a curve and zenith."""
    ax.fill_between(
        np.radians(az),
        90.0 - alt,
        np.zeros_like(az),
        alpha=alpha,
        color=color,
        linewidth=0,
    )


def make_sky_plot() -> matplotlib.axes.Axes:
    """Set up an azimuth-altitude polar plot of the sky.

    Some of the code here was copied from Astroplan's sky_plot()
    https://github.com/astropy/astroplan/blob/master/astroplan/plots/sky.py
    which is published with the following license:

    Copyright (c) 2015-2017, Astroplan Developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the documentation and/or
      other materials provided with the distribution.
    * Neither the name of the Astropy Team nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    plt.style.use('dark_background')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.canvas.manager.set_window_title('Skyplot')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlim(1, 91)
    ax.grid(True, which='major')

    degree_sign = u'\N{DEGREE SIGN}'

    # For positively-increasing range (e.g., range(1, 90, 15)),
    # labels go from middle to outside.
    r_labels = [
        '90' + degree_sign,
        '',
        '60' + degree_sign,
        '',
        '30' + degree_sign,
        '',
        '0' + degree_sign + ' Alt.',
    ]

    theta_labels = []
    for chunk in range(0, 8):
        label_angle = chunk*45.0
        while label_angle >= 360.0:
            label_angle -= 360.0
        if chunk == 0:
            theta_labels.append(f'N\n{label_angle:.0f}{degree_sign} Az')
        elif chunk == 2:
            theta_labels.append(f'E\n{label_angle:.0f}{degree_sign}')
        elif chunk == 4:
            theta_labels.append(f'S\n{label_angle:.0f}{degree_sign}')
        elif chunk == 6:
            theta_labels.append(f'W\n{label_angle:.0f}{degree_sign}')
        else:
            theta_labels.append(f'{label_angle:.0f}{degree_sign}')

    # Set ticks and labels.
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45, color='black')
    ax.set_thetagrids(range(0, 360, 45), theta_labels, color='#bbbbbb')

    return ax


def plot_tle(
        ax: matplotlib.axes.Axes,
        tle_filename: str,
        location: EarthLocation,
        time_start: datetime,
        time_stop: datetime,
    ) -> Tuple[datetime, datetime]:
    """Plot the trajectory of the first above-horizon pass within a specified time window."""

    tle = []
    with open(tle_filename) as tlefile:
        for line in tlefile:
            tle.append(line)

    target = ephem.readtle(tle[0], tle[1], tle[2])

    # Create PyEphem Observer object with location information
    observer = ephem.Observer()
    observer.lat = location.lat.rad
    observer.lon = location.lon.rad
    observer.elevation = location.height.to_value(u.m)

    az = []
    alt = []
    time_rise = None
    time_set = None
    t = time_start
    while t < time_stop:
        observer.date = ephem.Date(t)
        target.compute(observer)
        if target.alt >= 0:
            if len(az) == 0:
                time_rise = t
            az.append(target.az)
            alt.append(target.alt)
        elif time_rise is not None:
            break
        t += timedelta(seconds=1)

    if len(az) > 0:
        time_set = t
        az = np.degrees(np.array(az))
        alt = np.degrees(np.array(alt))
        plot_trajectory(ax, az, alt, color='black', label='TLE')

    return time_rise, time_set


def reachable_zone_scatter(
        ax: matplotlib.axes.Axes,
        mount_model: MountModel,
        axis_0_west_limit: float = 110,
        axis_0_east_limit: float = 110,
    ) -> None:
    """Generate a scatter plot showing the reachable zone.

    This function assumes an equatorial mount with limits on the right ascension axis.

    Args:
        ax: Axes object this function will plot on. This should be generated by `make_sky_plot()`.
        mount_model: Mount model from which this plot will be generated.
        axis_0_west_limit: Western limit on axis 0 in degrees from the meridian.
        axis_0_east_limit: Eastern limit on axis 0 in degrees from the meridian.
    """
    # convert from arg values to encoder position angles
    axis_0_west_limit = 180 - axis_0_west_limit
    axis_0_east_limit = 180 + axis_0_east_limit

    for meridian_side in MeridianSide:

        axis_0 = np.linspace(axis_0_west_limit, axis_0_east_limit, 20)
        if meridian_side == MeridianSide.EAST:
            axis_1 = np.linspace(0, 180, 20)
        else:
            axis_1 = np.linspace(180, 360, 20)
        ax0, ax1 = np.meshgrid(axis_0, axis_1)
        points = np.vstack([ax0.ravel(), ax1.ravel()])

        az = []
        alt = []
        for idx in range(points.shape[1]):
            topo = mount_model.encoders_to_topocentric(
                MountEncoderPositions(
                    Longitude(points[0][idx]*u.deg),
                    Longitude(points[1][idx]*u.deg),
                )
            )
            az.append(topo.az.deg)
            alt.append(topo.alt.deg)
        az = np.array(az)
        alt = np.array(alt)

        ax.scatter(np.radians(az), 90.0 - alt, label=meridian_side.name.title())


def plot_reachable_zone(
        ax: matplotlib.axes.Axes,
        mount_model: MountModel,
        axis_0_west_limit: float = 110,
        axis_0_east_limit: float = 110,
    ) -> None:
    """Plot area(s) of sky reachable by the mount.

    This only accounts for what area of the sky is reachable. It does not take into account whether
    the mount can keep up with a specific target's trajectory. Some trajectories, especially
    those that pass near the mount pole, may require slew rates that exceed what the mount is
    capable of.

    This function assumes an equatorial mount with limits on the right ascension axis.

    Args:
        ax: Axes object this function will plot on. This should be generated by `make_sky_plot()`.
        mount_model: Mount model from which this plot will be generated.
        axis_0_west_limit: Western limit on axis 0 in degrees from the meridian.
        axis_0_east_limit: Eastern limit on axis 0 in degrees from the meridian.
    """
    if axis_0_west_limit < 90 or axis_0_east_limit < 90:
        # current logic would not shade the correct regions of the polar plot
        raise ValueError('Axis limits less than 90 degrees from meridian are not supported')

    # convert from arg values to encoder position angles
    axis_0_west_limit = 180 - axis_0_west_limit
    axis_0_east_limit = 180 + axis_0_east_limit

    # place a dot at the position of the mount pole
    mount_pole_topo = mount_model.spherical_to_topocentric(
        UnitSphericalRepresentation(
            lon=0*u.deg,
            lat=90*u.deg,
        )
    )
    ax.plot(np.radians(mount_pole_topo.az.deg), 90.0 - mount_pole_topo.alt.deg, 'k.')

    # start with a white background in the polar plot area since the filled area colors look better
    # alpha-blended with a white background versus a black background
    fill_to_horizon(ax, np.linspace(0, 360, 100), 90*np.ones(100), color='#bbbbbb', alpha=1)

    alpha = 0.6

    for meridian_side in MeridianSide:
        if meridian_side == MeridianSide.EAST:
            color = 'tab:blue'
            legend_label = 'east of mount meridian'
            axis_1_range = np.linspace(0, 180, 100) + mount_model.model_params.axis_1_offset.deg
            az = np.linspace(mount_pole_topo.az.deg, mount_pole_topo.az.deg + 180, 100)
        else:
            color = 'tab:red'
            legend_label = 'west of mount meridian'
            axis_1_range = np.linspace(180, 360, 100) + mount_model.model_params.axis_1_offset.deg
            az = np.linspace(mount_pole_topo.az.deg - 180, mount_pole_topo.az.deg, 100)

        # add a circle patch outside the visible area of the plot purely for the purpose of
        # generating an entry in the legend for this region
        ax.add_patch(Circle((0, 100), radius=0, color=color, alpha=alpha, label=legend_label))

        alt = 90*np.ones_like(az)
        fill_to_horizon(ax, az, alt, color=color, alpha=alpha)

        for axis_0 in (axis_0_west_limit, axis_0_east_limit):
            az = []
            alt = []
            for axis_1 in axis_1_range:
                topo = mount_model.encoders_to_topocentric(
                    MountEncoderPositions(
                        Longitude(axis_0*u.deg),
                        Longitude(axis_1*u.deg),
                    )
                )
                az.append(topo.az.deg)
                alt.append(topo.alt.deg)
            az = np.array(az)
            alt = np.array(alt)
            ax.plot(np.radians(az), 90.0 - alt, ':', color='black')

            if axis_0 == axis_0_east_limit and meridian_side == MeridianSide.EAST:
                fill_to_horizon(ax, az, alt, color=color, alpha=alpha)
            elif axis_0 == axis_0_west_limit and meridian_side == MeridianSide.EAST:
                fill_to_zenith(ax, az, alt, color=color, alpha=alpha)
            elif axis_0 == axis_0_east_limit and meridian_side == MeridianSide.WEST:
                fill_to_zenith(ax, az, alt, color=color, alpha=alpha)
            elif axis_0 == axis_0_west_limit and meridian_side == MeridianSide.WEST:
                fill_to_horizon(ax, az, alt, color=color, alpha=alpha)


def plot_mount_motion(
        ax: matplotlib.axes.Axes,
        time_start: datetime,
        time_stop: datetime,
    ) -> None:
    """Plot curve showing position of the mount versus time from telemetry.

    Args:
        ax: Axes object on which to plot.
        time_start: Beginning of time interval to plot.
        time_stop: End of time interval to plot.
    """

    client = telem.open_client()
    query_api = client.query_api()
    df = query_api.query_data_frame(
        'from(bucket: "telem")'
        f'|> range(start: {int(time_start.timestamp())}, stop: {int(time_stop.timestamp())})'
        '|> filter(fn: (r) => r._measurement == "mount_position" '
            'and (r._field == "azimuth" or r._field == "altitude"))'
        '|> aggregateWindow(every: 1s, fn: first, createEmpty: false)'
    )

    if not df.empty:
        # pylint: disable=protected-access
        plot_trajectory(
            ax,
            az=df[df._field == 'azimuth']._value.values,
            alt=df[df._field == 'altitude']._value.values,
            label='Mount Telemetry'
        )


def main():
    """See module docstring at the top of this file"""

    parser = ArgParser(description='Make a plot of the sky to plan and evaluate tracking')
    parser.add_argument(
        '--tle',
        help='filename of two-line element (TLE) target ephemeris'
    )
    parser.add_argument(
        '--time-start',
        help=('UTC start time of 24-hour window in which the first above-horizon pass will be '
            ' plotted. Many natural language date formats are supported. If not specified, the '
            'current time will be used.'
        ),
        type=str)
    parser.add_argument(
        '--axis-west-limit',
        help='western limit for the right ascension axis in degrees from the meridian',
        default=110,
        type=float)
    parser.add_argument(
        '--axis-east-limit',
        help='eastern limit for the right ascension axis in degrees from the meridian',
        default=110,
        type=float)
    custom_model_group = parser.add_argument_group(
        title='Custom Mount Model Options',
        description=('Set all of these to use a custom mount model instead of a stored model. '
            'If no GPS is connected, also set the Observer Location Options.'),
    )
    custom_model_group.add_argument(
        '--mount-pole-az',
        help='azimuth of the mount pole',
        type=float)
    custom_model_group.add_argument(
        '--mount-pole-alt',
        help='altitude of the mount pole',
        type=float)
    gps_client.add_program_arguments(
        parser=parser,
        group_description=(
            'Set all of these to indicate observer location with a custom mount model'
        ),
    )
    args = parser.parse_args()
    custom_model_args = (args.mount_pole_az, args.mount_pole_alt)
    observer_args = (args.lat, args.lon, args.elevation)

    if args.time_start is not None:
        time_start = dateutil.parser.parse(args.time_start)
        time_start = time_start.replace(tzinfo=dateutil.tz.tzutc())
    else:
        time_start = datetime.utcnow()
    time_stop = time_start + timedelta(days=1)

    if all(arg is None for arg in custom_model_args):
        if any(arg is not None for arg in observer_args):
            # pylint: disable=not-callable
            parser.error('Observer location options require the custom mount model args to be set')
        mount_model = track.model.load_stored_model(max_age=None)
    elif any(arg is None for arg in custom_model_args):
        # pylint: disable=not-callable
        parser.error("Must give all args in the custom mount model group or none of them.")
    else:
        # Generate a custom mount model from program arguments
        location = gps_client.make_location_from_args(args)
        mount_model = track.model.load_default_model(
            mount_pole_az=Longitude(args.mount_pole_az*u.deg),
            mount_pole_alt=Longitude(args.mount_pole_alt*u.deg),
            location=location,
        )

    ax = make_sky_plot()
    plot_reachable_zone(
        ax,
        mount_model,
        axis_0_west_limit=args.axis_west_limit,
        axis_0_east_limit=args.axis_east_limit
    )

    if args.tle:
        print('Searching for first pass within the following time range:')
        print(time_start.isoformat() + 'Z')
        print(time_stop.isoformat() + 'Z')

        time_rise, time_set = plot_tle(
            ax,
            args.tle,
            mount_model.location,
            time_start,
            time_stop
        )
        if all((time_rise, time_set)):
            plot_mount_motion(ax, time_rise, time_set)
            print('Found a pass with these start and end times:')
            print(time_rise.isoformat() + 'Z')
            print(time_set.isoformat() + 'Z')
        else:
            print('No pass was found.')

    plt.legend(
        bbox_to_anchor=(0.8, 1.4),
        loc='upper left',
        borderaxespad=0,
        facecolor='#bbbbbb',
        framealpha=1,
        labelcolor='black',
    )
    # Shrink current axis by 20% so the legend will fit inside the figure window because matplotlib
    # developers apparently don't use their own code
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    plt.show()

if __name__ == "__main__":
    main()
