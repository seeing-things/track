#!/usr/bin/env python3

"""Topocentric plots of the sky for planning and post-mortem debugging.

This file can generate plots of the sky showing information that is relevant to tracking of various
object but especially satellite passes. Information includes regions of the sky reachable by the
mount within axis limits, trajectory of satellite pass from TLE, and trajectory of the mount from a
completed (or even in-progress) pass.
"""

from typing import Tuple, Optional
from configargparse import ArgParser
from datetime import datetime, timedelta
import dateutil
import requests
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import ephem
from astropy import units as u
from astropy.coordinates import Longitude, UnitSphericalRepresentation, EarthLocation
from influxdb import InfluxDBClient
import track
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
        alpha: float = 0.2,
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
        alpha: float = 0.2,
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

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar',})
    ax = plt.gca(projection='polar')
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
    for chunk in range(0, 7):
        label_angle = chunk*45.0
        while label_angle >= 360.0:
            label_angle -= 360.0
        if chunk == 0:
            theta_labels.append('N ' + '\n' + str(label_angle) + degree_sign
                                + ' Az')
        elif chunk == 2:
            theta_labels.append('E' + '\n' + str(label_angle) + degree_sign)
        elif chunk == 4:
            theta_labels.append('S' + '\n' + str(label_angle) + degree_sign)
        elif chunk == 6:
            theta_labels.append('W' + '\n' + str(label_angle) + degree_sign)
        else:
            theta_labels.append(str(label_angle) + degree_sign)

    # Set ticks and labels.
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
    ax.set_thetagrids(range(0, 360, 45), theta_labels)

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

    for meridian_side in MeridianSide:
        if meridian_side == MeridianSide.EAST:
            color = 'blue'
            legend_label = 'east of mount meridian'
            axis_1_range = np.linspace(0, 180, 100) + mount_model.model_params.axis_1_offset.deg
            az = np.linspace(mount_pole_topo.az.deg, mount_pole_topo.az.deg + 180, 100)
        else:
            axis_1_range = np.linspace(180, 360, 100) + mount_model.model_params.axis_1_offset.deg
            color = 'red'
            legend_label = 'west of mount meridian'
            az = np.linspace(mount_pole_topo.az.deg - 180, mount_pole_topo.az.deg, 100)

        # add a circle patch outside the visible area of the plot purely for the purpose of
        # generating an entry in the legend for this region
        ax.add_patch(Circle((0, 100), radius=0, color=color, alpha=0.2, label=legend_label))

        alt = 90*np.ones_like(az)
        fill_to_horizon(ax, az, alt, color=color)

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
            ax.plot(np.radians(az), 90.0 - alt, ':', color=color)

            if axis_0 == axis_0_east_limit and meridian_side == MeridianSide.EAST:
                fill_to_horizon(ax, az, alt, color=color)
            elif axis_0 == axis_0_west_limit and meridian_side == MeridianSide.EAST:
                fill_to_zenith(ax, az, alt, color=color)
            elif axis_0 == axis_0_east_limit and meridian_side == MeridianSide.WEST:
                fill_to_zenith(ax, az, alt, color=color)
            elif axis_0 == axis_0_west_limit and meridian_side == MeridianSide.WEST:
                fill_to_horizon(ax, az, alt, color=color)


def plot_mount_motion(
        ax: matplotlib.axes.Axes,
        mount_model: MountModel,
        time_start: datetime,
        time_stop: datetime
    ) -> None:
    """Plot curve showing position of the mount versus time from telemetry.

    Args:
        ax: Axes object on which to plot.
        mount_model: Mount model corresponding to the telemetry time period. Using the wrong model
            will produce invalid results.
        time_start: Beginning of time interval to plot.
        time_stop: End of time interval to plot.
    """

    db = InfluxDBClient(host='localhost', port=8086, database='telem')
    query = db.query(
        f'select mount_enc_0, mount_enc_1 from tracker where '
        f'time > {int(time_start.timestamp())}s and time < {int(time_stop.timestamp())}s'
    )
    if not query:
        # no telemetry exists for this time interval
        return
    df = pd.DataFrame(query.get_points())
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # downsample telemetry to at most 1 position per second of time
    df = df.resample('1s').first().dropna()

    az = []
    alt = []
    for _, row in df.iterrows():
        topo = mount_model.encoders_to_topocentric(
            MountEncoderPositions(
                Longitude(row['mount_enc_0']*u.deg),
                Longitude(row['mount_enc_1']*u.deg),
            )
        )
        az.append(topo.az.deg)
        alt.append(topo.alt.deg)
    az = np.array(az)
    alt = np.array(alt)
    if np.any(alt > 0):
        plot_trajectory(ax, az, alt, label='Mount Telemetry')


def main():

    parser = ArgParser(description='Make a plot of the sky to plan and evaluate tracking')
    parser.add_argument(
        '--tle-filename',
        help='filename of two-line element (TLE) target ephemeris'
    )
    parser.add_argument(
        '--time-start',
        help=('UTC start time of 24-hour window in which the first above-horizon pass will be '
            ' plotted. Many natural language date formats are supported. If not specified, the '
            'current time will be used.'
        ),
        type=str)
    args = parser.parse_args()

    if args.time_start is not None:
        time_start = dateutil.parser.parse(args.time_start)
        time_start = time_start.replace(tzinfo=dateutil.tz.tzutc())
    else:
        time_start = datetime.utcnow()
    time_stop = time_start + timedelta(days=1)

    mount_model = track.model.load_stored_model(max_age=None)

    ax = make_sky_plot()
    plot_reachable_zone(ax, mount_model)

    if args.tle_filename:
        print('Searching for first pass within the following time range:')
        print(time_start.isoformat() + 'Z')
        print(time_stop.isoformat() + 'Z')

        time_rise, time_set = plot_tle(
            ax,
            args.tle_filename,
            mount_model.location,
            time_start,
            time_stop
        )
        if all((time_rise, time_set)):
            plot_mount_motion(ax, mount_model, time_rise, time_set)
            print('Found a pass with these start and end times:')
            print(time_rise.isoformat() + 'Z')
            print(time_set.isoformat() + 'Z')
        else:
            print('No pass was found.')

    plt.legend(
        bbox_to_anchor=(0.8, 1.4),
        loc='upper left',
        borderaxespad=0,
    )
    # Shrink current axis by 20% so the legend will fit inside the figure window because matplotlib
    # developers apparently don't use their own code
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    plt.show()

if __name__ == "__main__":
    main()
