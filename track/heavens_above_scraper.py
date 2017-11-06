#!/usr/bin/env python

from bs4 import BeautifulSoup
import urllib2
import re
import track

def urlify(s):

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single underscore
    s = re.sub(r"\s+", '_', s)

    return s

def main():
    parser = track.ArgParser()
    parser.add_argument('--lat', required=True, help='latitude of observer (+N)')
    parser.add_argument('--lon', required=True, help='longitude of observer (+E)')
    parser.add_argument('--elevation', required=True, help='elevation of observer (m)', type=float)
    args = parser.parse_args()

    base_url = 'http://www.heavens-above.com/'

    bright_sats_url = base_url + 'AllSats.aspx?lat={}&lng={}&alt={}'.format(args.lat, args.lon, args.elevation)

    bright_sats_page = urllib2.urlopen(bright_sats_url).read()
    bright_sats_soup = BeautifulSoup(bright_sats_page, 'lxml')

    # find the rows in the table listing the satellite passes
    table = bright_sats_soup.find_all('table')[4]
    table_body = table.find_all('tbody')[0]
    rows = table_body.find_all('tr')

    tles = []
    for row in rows:

        # extract the satellite id and name from this table row
        onclick_str = row['onclick']
        url_suffix = re.findall(r"'([^']*)'", onclick_str)[0]
        satid = re.findall(r"satid=([0-9]*)", url_suffix)[0]
        satname = row.find_all('td')[0].string
        pass_detail_url = base_url + url_suffix

        print('Getting TLE for ' + satname + '...')

        # get the TLE from the orbit details page for this satellite
        orbit_url = base_url + 'orbit.aspx?satid=' + satid
        orbit_page = urllib2.urlopen(orbit_url).read()
        orbit_soup = BeautifulSoup(orbit_page, 'lxml')
        pre_tag = orbit_soup.find_all('pre')[0]
        span_tags = pre_tag.find_all('span')
        tle = [satname]
        for span_tag in span_tags:
            assert span_tag['id'].startswith('ctl00_cph1_lblLine')
            tle.append(span_tag.string)
        tles.append(tle)

        with open('/tmp/tles/' + urlify(satname) + '.tle', 'w') as f:
            for line in tle:
                f.write(line + '\n')

        if len(tles) > 2:
            break

if __name__ == "__main__":
    main()
