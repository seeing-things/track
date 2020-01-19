import os
import tempfile
import subprocess
from typing import Optional
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import wcs
from astropy import units as u

class NoSolutionException(Exception):
    """Raised when plate solving fails to find a solution."""

def plate_solve(frame: np.ndarray, camera_width: Optional[float] = None) -> SkyCoord:
    """Perform plate solving on a camera frame using Astrometry.net software.

    This function requires astrometry.net software to be installed and the solve-field binary to be
    on the PATH.

    Args:
        frame: A numpy array containing a grayscale image of the sky. For a successful solution
            at least four stars must be detectable in the frame.
        camera_width: Width of the camera field of view in degrees. This information greatly
            reduces the time required to find a solution since the search space is reduced.

    Returns:
        A SkyCoord object containing the equatorial coordinates for the center of the frame.

    Raises:
        NoSolutionException when a solution could not be found.
    """

    # Must pass frame to astrometry.net as a file and read results from files, so do this in a
    # unique temporary directory that is deleted as soon as we are done using it
    with tempfile.TemporaryDirectory() as tempdir:

        filename_prefix = 'guidescope_frame'
        frame_filename = os.path.join(tempdir, filename_prefix + '.fits')

        hdu = fits.PrimaryHDU(frame)
        hdu.writeto(frame_filename, overwrite=True)

        args = [
            'solve-field',
            '--overwrite',
            '--objs=100',
            '--depth=20',
            '--no-plots',
        ]
        if camera_width is not None:
            args.append('--scale-low={:.2f}'.format(camera_width - 0.1))
            args.append('--scale-high={:.2f}'.format(camera_width + 0.1))
        args.append(frame_filename)

        # Call astrometry.net binary solve-field
        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            wcs_file = fits.open(os.path.join(tempdir, filename_prefix + '.wcs'))
        except FileNotFoundError:
            raise NoSolutionException()

        # get "world coordinates" for center of frame
        wcs_header = wcs.WCS(header=wcs_file[0].header)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        center_coord = wcs_header.all_pix2world(
            (frame_width - 1) / 2.0,
            (frame_height - 1) / 2.0,
            0
        )

        # making an assumption here that the coordinates reported by astrometry.net are actually in
        # ICRS frame or something close enough to this
        return SkyCoord(center_coord[0] * u.deg, center_coord[1] * u.deg, frame='icrs')
