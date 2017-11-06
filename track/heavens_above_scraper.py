#!/usr/bin/env python

from bs4 import BeautifulSoup
import requests
import re
import track
import datetime
import monthdelta
import os

def urlify(s):

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single underscore
    s = re.sub(r"\s+", '_', s)

    return s

# convert a given date object to the number-of-months-since-0-AD, because that's how HA encodes it
def date_to_monthnum(date):
    ad1 = datetime.date(1, 1, 1)
    return (monthdelta.monthmod(ad1, date)[0].months + 12)

def print_tz_help():
    tz_soup = BeautifulSoup(requests.get('http://heavens-above.com/SelectLocation.aspx').text, 'lxml')

    # find the dropdown box containing the tz code to description mappings
    options = tz_soup.find('select', {'name': 'ctl00$cph1$listTimeZones'}).find_all('option')

    print('{:13s} {}'.format('CODE', 'DESCRIPTION'))
    for option in options:
        print('{:13s} {}'.format(option['value'], option.string))

def main():
    parser = track.ArgParser()
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
    parser.add_argument('--tz', required=True, help='time zone short code (use \'help\' for a list of codes)')
    parser.add_argument('--year', required=False, help='observation year (default: now)', type=int)
    parser.add_argument('--month', required=False, help='observation month (default: now)', type=int)
    parser.add_argument('--day', required=False, help='observation day-of-month (default: now)', type=int)
    parser.add_argument('--ampm', required=False, help='morning or evening (\'AM\' or \'PM\'; default: \'PM\')', default='PM')
    parser.add_argument('--mag-limit', required=True, help='magnitude cutoff for object passes', type=float)

    args = parser.parse_args()

    if args.tz == 'help':
        print_tz_help()
        return

    if args.ampm.upper() != 'AM' and args.ampm.upper() != 'PM':
        raise Exception('The AM/PM argument can only be \'AM\' or \'PM\'.')

    if args.year is not None and args.month is not None and args.day is not None:
        when = datetime.datetime(args.year, args.month, args.day).date()
    elif args.year is None and args.month is None and args.day is None:
        when = datetime.datetime.now().date()
    else:
        raise Exception('If an explicit observation date is given, then year, month, and day must all be specified.')

    base_url = 'http://www.heavens-above.com/'

    bright_sats_url = base_url + 'AllSats.aspx?lat={}&lng={}&alt={}&tz={}'.format(args.lat, args.lon, args.elevation, args.tz)

    # do an initial page request so we can get the __VIEWSTATE and __VIEWSTATEGENERATOR values
    pre_soup = BeautifulSoup(requests.get(bright_sats_url).text, 'lxml')
    view_state           = pre_soup.find('input', {'type': 'hidden', 'name': '__VIEWSTATE'         })['value']
    view_state_generator = pre_soup.find('input', {'type': 'hidden', 'name': '__VIEWSTATEGENERATOR'})['value']

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

    tles = []
    for row in rows:

        cols = row.find_all('td')

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

        # extract the satellite id and name from this table row
        onclick_str = row['onclick']
        url_suffix = re.findall(r"'([^']*)'", onclick_str)[0]
        satid = re.findall(r"satid=([0-9]*)", url_suffix)[0]
        pass_detail_url = base_url + url_suffix

        print('Getting TLE for ' + sat + '...')

        # get the TLE from the orbit details page for this satellite
        orbit_url = base_url + 'orbit.aspx?satid=' + satid
        orbit_page = requests.get(orbit_url).text
        orbit_soup = BeautifulSoup(orbit_page, 'lxml')
        pre_tag = orbit_soup.pre
        span_tags = pre_tag.find_all('span')
        tle = [sat]
        for span_tag in span_tags:
            assert span_tag['id'].startswith('ctl00_cph1_lblLine')
            tle.append(span_tag.string)
        tles.append(tle)

        filename = os.path.join(os.path.normpath(args.outdir), urlify(sat) + '.tle')
        with open(filename, 'w') as f:
            for line in tle:
                f.write(line + '\n')

        if len(tles) > 2:
            break

if __name__ == "__main__":
    main()
