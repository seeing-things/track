#!/usr/bin/env python3

"""
Automatic focus script and associated algorithms.
"""

import time
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from track import cameras, focusers
from track.config import ArgParser


def create_circular_mask(
        width: int,
        height: int,
        radius: float,
        center: Tuple[float, float]
    ) -> np.ndarray:
    """Create a circular mask.

    Args:
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        radius: Radius of the mask.
        center: Coordinates of the center of the mask.

    Returns:
        An array containing a circular mask.
    """
    radius_squared = radius**2
    y_grid, x_grid = np.ogrid[:height, :width]
    dist_from_center_squared = (y_grid - center[0])**2 + (x_grid - center[1])**2
    return dist_from_center_squared <= radius_squared


def estimate_hfr(
        image: np.ndarray,
        hfr_max: Optional[float] = None,
        tolerance: float = 0.1
    ) -> float:
    """Estimates the half flux radius (HFR) of a star.

    Args:
        image: A greyscale image containing a star. Background and other stars should be removed.
        hfr_max: Maximum radius in pixels. No check is performed to ensure that this value is large
            enough.
        tolerance: Max allowed error in the estimate in pixels.

    Returns:
        The estimated half flux radius in pixels. Will be in the range [0, hfr_max].
    """
    height, width = image.shape
    center_of_mass = ndimage.center_of_mass(image)
    total_flux = np.sum(image)
    half_flux = total_flux / 2

    hfr_min = 0.0
    if hfr_max is None:
        # a circle with radius equal to the diagonal of the image will always encompass the entire
        # image no matter where the center is located
        hfr_max = np.sqrt(width**2 + height**2)

    while True:
        hfr = (hfr_min + hfr_max) / 2
        if hfr_max - hfr_min < tolerance:
            break
        mask = create_circular_mask(width, height, hfr, center_of_mass)
        flux = np.sum(image * mask)
        if flux > half_flux:
            hfr_max = hfr
        else:
            hfr_min = hfr

    return hfr


# TODO: Not using this exception right now
class FocusEstimatorException(Exception):
    """Something went wrong trying to estimate ideal focus position."""


def estimate_ideal_focuser_position(focuser_steps: np.ndarray, hfrs: np.ndarray) -> int:
    """Estimate ideal focuser position.

    Estimates the ideal focuser position by fitting a hyperbolic function to a set of HFR estimates
    and solves for the focuser position that minimizes the HFR.

    Args:
        focuser_steps: Array of focuser positions.
        hfrs: Array of half-flux radii corresponding to the focuser positions.

    Returns:
        Estimated ideal focuser position.
    """
    def v_curve(x, alpha, beta, x_0, y_0):
        return beta * np.sqrt((x - x_0)**2 + alpha**2) + y_0

    # TODO: initial x_0 param value is specific to my focuser which has a position range of 4200
    # TODO: Add bounds on parameters
    popt, _ = curve_fit(v_curve, focuser_steps, hfrs, p0=(350, 1.75e-2, 2100, 0))

    return int(np.round(popt[2]))


def autofocus(
        camera: cameras.Camera,
        focuser: focusers.Focuser,
        focuser_steps: np.ndarray,
        skip_final_move: bool = False,
    ) -> int:
    """Automatically focus the camera.

    This algorithm estimates the half-flux radius (HFR) of a bright star across a set of focuser
    positions and estimates the position that minimizes the HFR, and then optionally moves the
    focuser to that position. The mount should be tracking a bright star such that it remains
    steady in the camera for the duration of the procedure.

    Args:
        camera: Camera attached to the optical system to be focused.
        focuser: The focuser to be adjusted.
        focuser_steps: A set of focuser positions to sample.
        skip_final_move: By default, this function will move the focuser to the estimated ideal
            position. If this is True, that final move will be skipped and the focuser will be left
            at the last position in `focuser_steps`.

    Returns:
        The estimated ideal focuser position.
    """
    hfrs = np.zeros(focuser_steps.size)
    for idx, position in enumerate(focuser_steps):
        print(
            f'Estimating HFR at focuser position {position:4d} '
            f'({idx:3d} of {focuser_steps.size:3d})...',
            end='',
            flush=True)
        focuser.target_position = position
        focuser.move_to_target_position(blocking=True)
        image = camera.get_frame()

        # Reject background
        image[image < 0.1*np.max(image)] = 0

        hfrs[idx] = estimate_hfr(image)
        print('done.')

    ideal_position = estimate_ideal_focuser_position(focuser_steps, hfrs)
    print(f'Estimated ideal focuser position: {ideal_position}')

    if not skip_final_move:
        print(f'Moving focuser to {ideal_position}...', end='', flush=True)
        focuser.target_position = ideal_position
        focuser.move_to_target_position(blocking=True)
        print('done.')

    return ideal_position


def main():
    """Run the full autofocus algorithm."""

    parser = ArgParser()
    cameras.add_program_arguments(parser, profile='align')
    focusers.add_program_arguments(parser)
    args = parser.parse_args()

    camera = cameras.make_camera_from_args(args, profile='align')
    focuser = focusers.make_focuser_from_args(args)
    focuser.motor_speed = 2  # fastest speed

    time_start = time.perf_counter()

    # Start by sparsely sampling the entire range to get a rough idea of where best focus is
    positions = np.linspace(focuser.min_position, focuser.max_position, 5, dtype=int)
    prelim_ideal_position = autofocus(camera, focuser, positions, skip_final_move=True)

    # Refine by taking a greater number of closely spaced samples centered on the first estimate
    positions = np.linspace(
        start=max(prelim_ideal_position - 200, focuser.min_position),
        stop=min(prelim_ideal_position + 200, focuser.max_position),
        num=20,
        dtype=int)
    autofocus(camera, focuser, positions)

    time_elapsed = time.perf_counter() - time_start
    print(f'Elapsed time: {time_elapsed} s')


if __name__ == "__main__":
    main()
