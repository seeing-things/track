#!/usr/bin/env python3

"""NTP daemon status check utility.

This module provides a function that checks the status of the local NTP daemon to ensure that the
system clock is synchronized to a reliable peer with an error below acceptable limits.
"""

from io import StringIO
import subprocess
import pandas as pd
from track.config import ArgParser


class NTPCheckFailure(Exception):
    """Raised when NTP status check fails"""


def check_ntp_status(
        gps_max_offset: float = 0.5,
        pps_max_offset: float = 0.1,
        max_age: float = 60.0
    ):
    """Checks NTP daemon status and raises an exception if any checks fail.

    Success criteria:
    - GPS and PPS peers exist
    - PPS is the selected system peer
    - GPS offset magnitude is less than the value specified in the `gps_max_offset` argument
    - PPS offset magnitude is less than the value specified in the `pps_max_offset` argument
    - GPS and PPS last heard from within the last `max_age` seconds

    Args:
        gps_max_offset: Maximum allowed magnitude of the GPS peer offset in seconds
        pps_max_offset: Maximum allowed magnitude of the PPS peer offset in seconds
        max_age: Maximum time since GPS and PPS devices were last heard from in seconds

    Raises:
        NTPCheckFailure if any of the criteria are not met.
    """

    # The implementation relies on parsing the console output of the ntpq program since no obvious
    # alternative method was found. This unfortunately means that the implementation relies on
    # some specifics of the formatting of the peers table printed by ntpq.

    # See https://docs.ntpsec.org/latest/ntpq.html for details on the output of ntpq
    completed_process = subprocess.run(['ntpq', '-p'], check=True, stdout=subprocess.PIPE)
    ntpq_output = completed_process.stdout.decode()
    data = StringIO(ntpq_output)
    df = pd.read_csv(data, delim_whitespace=True, comment='=')

    gps_row = df[df.refid == '.GPS.']
    pps_row = df[df.refid == '.PPS.']

    if gps_row.empty:
        raise NTPCheckFailure('ntpd does not know about the GPS receiver')

    if pps_row.empty:
        raise NTPCheckFailure('ntpd does not know about the GPS receiver PPS input')

    if not pps_row.remote.item().startswith('*'):
        raise NTPCheckFailure('PPS is not the selected NTP system peer')

    if 1e-3*gps_row.offset.item() > gps_max_offset:
        raise NTPCheckFailure(f'GPS offset {gps_row.offset.item()} ms exceeds '
            f'{1e3*gps_max_offset} ms limit')

    if 1e-3*pps_row.offset.item() > pps_max_offset:
        raise NTPCheckFailure(f'PPS offset {pps_row.offset.item()} ms exceeds '
            f'{1e3*pps_max_offset} ms limit')

    if int(gps_row.when.item()) > max_age:
        raise NTPCheckFailure(f'GPS last heard from {gps_row.when.item()} seconds ago '
            f'(exceeds {max_age} s limit)')

    if int(pps_row.when.item()) > max_age:
        raise NTPCheckFailure(f'PPS last heard from {pps_row.when.item()} seconds ago '
            f'(exceeds {max_age} s limit)')


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments associated with this module.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    ntp_group = parser.add_argument_group(
        title='NTP Options',
        description='Options that apply to NTP time synchronization',
    )
    ntp_group.add_argument(
        '--ignore-ntp-check',
        help='ignore NTP check failures',
        action='store_true',
    )


if __name__ == "__main__":
    check_ntp_status()
    print('All NTP status checks passed')
