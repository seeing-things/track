#!/usr/bin/env python3

"""Scrapes the Heavens Above website (heavens-above.com) to download TLEs for nightly passes.

Heavens Above predicts which satellites will be visible at a given location and on a particular
morning or evening. This program scrapes that website in order to download a set of fresh TLE files
for such satellites.
"""

import os
import re
from math import inf
import datetime
import monthdelta
import requests
from bs4 import BeautifulSoup
from astropy.coordinates import EarthLocation
import astropy.units as u
import track
from track.gps_client import GPSValues, GPSMargins, GPS


def urlify(s: str) -> str:
    """Transform a string into a string that is a valid part of a URL"""

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single underscore
    s = re.sub(r"\s+", '_', s)

    return s


def date_to_monthnum(date: datetime.date) -> int:
    """Convert a given date object to the number-of-months-since-0-AD.

    Why? Because that's how Heavens Above encodes it.

    Args:
        date: The date to convert.

    Returns:
        The number of months since 0 AD.
    """
    ad1 = datetime.date(1, 1, 1)
    return monthdelta.monthmod(ad1, date)[0].months + 12


def print_timezone_help():
    """Print help message for how to use the timezones 'tz' program argument"""
    tz_soup = BeautifulSoup(
        requests.get('http://heavens-above.com/SelectLocation.aspx').text,
        'lxml'
    )

    # find the dropdown box containing the tz code to description mappings
    options = tz_soup.find('select', {'name': 'ctl00$cph1$listTimeZones'}).find_all('option')

    print('{:13s} {}'.format('CODE', 'DESCRIPTION'))
    for option in options:
        print('{:13s} {}'.format(option['value'], option.string))


def main():
    """See module docstring at the top of this file."""

    parser = track.ArgParser()
    parser.add_argument(
        'outdir',
        help='output directory'
    )
    parser.add_argument(
        '--mag-limit',
        required=True,
        help='magnitude cutoff for object passes',
        type=float
    )
    observer_group = parser.add_argument_group(
        title='Observer Location Options',
        description='Setting all three of these options will override GPS',
    )
    observer_group.add_argument(
        '--lat',
        help='latitude of observer (+N)',
        type=float,
    )
    observer_group.add_argument(
        '--lon',
        help='longitude of observer (+E)',
        type=float,
    )
    observer_group.add_argument(
        '--elevation',
        help='elevation of observer (m)',
        type=float
    )
    time_group = parser.add_argument_group(
        title='Time Options',
        description='Options pertaining to time of observation',
    )
    time_group.add_argument(
        '--tz',
        required=True,
        help='time zone short code (use \'help\' for a list of codes)'
    )
    time_group.add_argument(
        '--year',
        required=False,
        help='observation year (default: now)',
        type=int
    )
    time_group.add_argument(
        '--month',
        required=False,
        help='observation month (default: now)',
        type=int
    )
    time_group.add_argument(
        '--day',
        required=False,
        help='observation day-of-month (default: now)',
        type=int
    )
    time_group.add_argument(
        '--ampm',
        required=False,
        help='morning or evening (\'AM\' or \'PM\')',
        default='PM'
    )
    args = parser.parse_args()

    if args.tz == 'help':
        print_timezone_help()
        return

    if args.ampm.upper() != 'AM' and args.ampm.upper() != 'PM':
        raise Exception('The AM/PM argument can only be \'AM\' or \'PM\'.')

    if args.year is not None and args.month is not None and args.day is not None:
        when = datetime.datetime(args.year, args.month, args.day).date()
    elif args.year is None and args.month is None and args.day is None:
        when = datetime.datetime.now().date()
    else:
        raise Exception('If an explicit observation date is given, then year, month, and day must '
            'all be specified.')

    # Get location of observer from arguments or from GPS
    if all(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        print('Location (lat, lon, elevation) specified by program args. This will override GPS.')
        location = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.elevation*u.m)
    elif any(arg is not None for arg in [args.lat, args.lon, args.elevation]):
        parser.error("Must give all of lat, lon, and elevation or none of them.")
    else:
        with GPS() as g:
            location = g.get_location(
                timeout=10.0,
                need_3d=True,
                err_max=GPSValues(
                    lat=100.0,
                    lon=100.0,
                    alt=100.0,
                    track=inf,
                    speed=inf,
                    climb=inf,
                    time=0.01
                ),
                margins=GPSMargins(speed=inf, climb=inf, time=1.0)
            )
            print(
                'Got location from GPS: '
                f'lat: {location.lat:.5f}, '
                f'lon: {location.lon:.5f}, '
                f'altitude: {location.height:.2f}'
            )

    base_url = 'http://www.heavens-above.com/'

    bright_sats_url = (base_url + f'AllSats.aspx?lat={location.lat.deg}&lng={location.lon.deg}'
        f'&alt={location.height.value}&tz={args.tz}')

    # do an initial page request so we can get the __VIEWSTATE and __VIEWSTATEGENERATOR values
    pre_soup = BeautifulSoup(requests.get(bright_sats_url).text, 'lxml')
    view_state = pre_soup.find(
        'input',
        {'type': 'hidden', 'name': '__VIEWSTATE'         }
    )['value']
    view_state_generator = pre_soup.find(
        'input',
        {'type': 'hidden', 'name': '__VIEWSTATEGENERATOR'}
    )['value']

    post_data = [
        ('__EVENTTARGET',                               ''),
        ('__EVENTARGUMENT',                             ''),
        ('__LASTFOCUS',                                 ''),
        ('__VIEWSTATE',                                 view_state),
        ('__VIEWSTATEGENERATOR',                        view_state_generator),
#        ('utcOffset',                                   '0'), # uhhhhhh...
        ('ctl00$ddlCulture',                            'en'),
        ('ctl00$cph1$TimeSelectionControl1$comboMonth', str(date_to_monthnum(when))),
        ('ctl00$cph1$TimeSelectionControl1$comboDay',   str(when.day)),
        ('ctl00$cph1$TimeSelectionControl1$radioAMPM',  args.ampm.upper()),
        ('ctl00$cph1$TimeSelectionControl1$btnSubmit',  'Update'),
        ('ctl00$cph1$radioButtonsMag',                  '5.0'),
    ]

    bright_sats_soup = BeautifulSoup(requests.post(bright_sats_url, data=post_data).text, 'lxml')

    # find the rows in the table listing the satellite passes
    table = bright_sats_soup.find('table', {'class': 'standardTable'})
    rows = table.tbody.find_all('tr')

    # this number is deceptive because it's pre-magnitude-filtering
    print('Found {} (pre-filtered) satellite passes.'.format(len(rows)))

    for row in rows:

        cols = row.find_all('td')

        # We're not using all of these now, but preserving them is useful as documentation
        # pylint: disable=unused-variable
        sat        =       cols[ 0].string
        mag        = float(cols[ 1].string)
        start_time =       cols[ 2].string # <-- in local time
        start_alt  =       cols[ 3].string
        start_az   =       cols[ 4].string
        high_time  =       cols[ 5].string # <-- in local time
        high_alt   =       cols[ 6].string
        high_az    =       cols[ 7].string
        end_time   =       cols[ 8].string # <-- in local time
        end_alt    =       cols[ 9].string
        end_az     =       cols[10].string

        # manually enforce the magnitude threshold
        if mag > args.mag_limit:
            continue

        # extract the satellite id from the onclick attribute of this table row
        onclick_str = row['onclick']
        url_suffix = re.findall(r"'([^']*)'", onclick_str)[0]
        satid = re.findall(r"satid=([0-9]*)", url_suffix)[0]

        print('Getting TLE for ' + sat + '...')

        # get the TLE from the orbit details page for this satellite
        orbit_url = base_url + 'orbit.aspx?satid=' + satid
        orbit_page = requests.get(orbit_url).text
        orbit_soup = BeautifulSoup(orbit_page, 'lxml')
        span_tags = orbit_soup.pre.find_all('span')
        assert len(span_tags) == 2
        tle = [sat]
        for span_tag in span_tags:
            assert span_tag['id'].startswith('ctl00_cph1_lblLine')
            tle.append(span_tag.string)

        os.makedirs(args.outdir, exist_ok=True)
        filename = os.path.join(os.path.normpath(args.outdir), urlify(sat) + '.tle')
        with open(filename, 'w') as f:
            for line in tle:
                f.write(line + '\n')


if __name__ == "__main__":
    main()
