#!/usr/bin/env python3

#import track
from gpsd import *

TIMEOUT = 10.0
NEED_3D = True
ERR_MAX = GPS.Errors(time=0.001, lat=1.0, lon=1.0, alt=1.0, track=1.0, speed=1.0, climb=1.0)

def main():
    with GPS() as g:
        try:
            loc = g.get_location(TIMEOUT, NEED_3D, ERR_MAX)
            print(loc)
        finally:
            print('fix_type: {}'.format(g.fix_type))
            print('location: {}'.format(g.location))
            print('errors:   {}'.format(g.errors))

if __name__ == '__main__':
    main()
