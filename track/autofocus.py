#!/usr/bin/env python3

"""
Automatic focus script and associated algorithms.
"""

import atexit
from datetime import datetime
import logging
import signal
import subprocess
import os
import time
from typing import Optional, Tuple
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from track import cameras, focusers, logs, targets
from track.config import ArgParser, CONFIG_PATH, DATA_PATH
from track.mounts import MeridianSide
from track.targets import TargetType


logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT_S = 2.0


def terminate_subprocess(process: subprocess.Popen) -> None:
    """Terminate a running subprocess"""
    logger.info('Terminating subprocess.')
    process.send_signal(signal.SIGINT)
    time_sigint = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - time_sigint > SHUTDOWN_TIMEOUT_S:
            logger.warning('Subprocess shutdown timeout; resorting to SIGKILL.')
            process.kill()
            break
        time.sleep(0.1)
    logger.info('Subprocess terminated.')


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


def v_curve(x: np.ndarray, alpha: float, beta: float, x_0: float, y_0: float) -> np.ndarray:
    """Hyperbolic model of focuser V-curve.

    Half flux radius (HFR) as a function of focuser position is approximately hyperbolic. Far from
    the ideal focus position, HFR is nearly linear in focus position but the bottom of the "V" is
    rounded off because even with ideal focus the star is not infintesimally small due to
    diffraction, the point spread function of the optics, and seeing conditions. Thus as the
    focuser is near the ideal position, these other effects become the dominant contributors to the
    HFR.

    The coordinates at the minimum of this function are (x_0, alpha * beta + y_0).

    Args:
        x: Array of x coordinates. In this application, x represents the focuser position.
        alpha: Affects the sharpness of the point of the V, where smaller values yield a sharper
            point. When this value is 0, the to sides of the curve are straight lines. The sign of
            this value has no impact on the curve shape.
        beta: The slope of the curve when x approaches infinity. In this focuser application beta
            will be a positive value, since a negative value would yield an upside-down V.
        x_0: The x coordinate of the tip of the V, which is the focuser position that minimizes the
            HFR.
        y_0: The y offset. When alpha equals 0, this is the y coordinate of the tip of the V. When
            alpha is nonzero, the straight lines tangent to the arms of the curve as x approaches
            positive and negative infinity intersect at (x_0, y_0).

    Returns:
        The function evaluated at the points in x.
    """
    return beta * np.sqrt((x - x_0)**2 + alpha**2) + y_0


def estimate_ideal_focuser_position(
        focuser_steps: np.ndarray,
        hfrs: np.ndarray,
        show_plot: bool = False
    ) -> int:
    """Estimate ideal focuser position.

    Estimates the ideal focuser position by fitting a hyperbolic function to a set of HFR estimates
    and solves for the focuser position that minimizes the HFR. The solution is constrained to
    fall within the same range as the provided array of focuser positions.

    Args:
        focuser_steps: Array of focuser positions.
        hfrs: Array of half-flux radii corresponding to the focuser positions.
        show_plot: If true, show a plot HFR versus focuser position data against fitted curve.

    Returns:
        Estimated ideal focuser position.

    Raises:
        RuntimeError if the curve fit fails.
    """
    position_mid = (np.max(focuser_steps) + np.min(focuser_steps)) / 2

    try:
    # pylint: disable=unbalanced-tuple-unpacking
        popt, _ = curve_fit(
            v_curve,
            focuser_steps,
            hfrs,
            p0=(1, 1, position_mid, 0),
            bounds=(
                (0, 0, np.min(focuser_steps), -np.inf),  # lower bounds on param values
                (np.inf, np.inf, np.max(focuser_steps), np.inf)  # upper bounds on param values
            )
        )
    except RuntimeError:
        logger.exception('curve_fit failed.')
        raise

    rms_error = np.sqrt(np.mean(np.square(v_curve(focuser_steps, *popt) - hfrs)))
    logger.info(f'Focuser V-curve solution RMS error: {rms_error} pixels')

    ideal_position = int(np.round(popt[2]))

    if show_plot:
        plt.figure()
        plt.plot(focuser_steps, hfrs, '.', label='HFR Estimates')
        plt.plot(focuser_steps, v_curve(focuser_steps, *popt), label='Curve Fit')
        plt.plot(ideal_position, v_curve(ideal_position, *popt), 'k.', label='Ideal Position')
        plt.xlabel('Focuser Position')
        plt.ylabel('Half Flux Radius [pixels]')
        plt.grid(True)
        plt.legend()
        plt.title('HFR vs Focuser Position (V-Curve)')
        plt.show()

    return ideal_position


def autofocus(
        camera: cameras.Camera,
        focuser: focusers.Focuser,
        focuser_steps: np.ndarray,
        skip_final_move: bool = False,
        output_dir: Optional[str] = None,
        show_plot: bool = False,
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
        output_dir: Directory in which to save images and a CSV file containing HFR and focuser
            position data for debugging purposes. By default no files are saved.
        show_plot: If true, show a plot HFR versus focuser position data against fitted curve.

    Returns:
        The estimated ideal focuser position.
    """
    hfrs = np.zeros(focuser_steps.size)
    for idx, position in enumerate(focuser_steps):
        focuser.target_position = position
        focuser.move_to_target_position(blocking=True)
        image = camera.get_frame()

        if output_dir is not None:
            iio.imwrite(
                os.path.join(
                    output_dir,
                    f'{datetime.utcnow().isoformat(timespec="seconds").replace(":", "")}_'
                    f'focuser_step_{position:04d}_'
                    f'exposure_{camera.exposure}_'
                    f'gain_{camera.gain}.png'
                ),
                image)

        # Reject background
        image[image < 0.1*np.max(image)] = 0

        hfrs[idx] = estimate_hfr(image)

        logger.info(
            f'HFR {hfrs[idx]:6.2f} pixels at position {position:4d} '
            f'({idx + 1:3d} of {focuser_steps.size:3d}).')

    if output_dir is not None:
        np.savetxt(
            os.path.join(
                output_dir,
                f'{datetime.utcnow().isoformat(timespec="seconds").replace(":", "")}_'
                'hfr_vs_focuser_position_data.csv'
            ),
            np.stack((focuser_steps, hfrs)).T, delimiter=',',
            fmt='%f',
            header='focuser position,half flux radius',
        )

    ideal_position = estimate_ideal_focuser_position(focuser_steps, hfrs, show_plot)
    logger.info(f'Estimated ideal focuser position: {ideal_position}')

    if not skip_final_move:
        logger.info(f'Moving focuser to {ideal_position}.')
        focuser.target_position = ideal_position
        focuser.move_to_target_position(blocking=True)

    return ideal_position


def main():
    """Run the full autofocus algorithm."""

    parser = ArgParser(additional_config_files=[os.path.join(CONFIG_PATH, 'autofocus.cfg')])
    parser.add_argument(
        '--meridian-side',
        help='side of meridian for equatorial mounts to prefer',
        required=True,
        type=str.upper,
        choices=tuple(m.name for m in MeridianSide),
    )
    parser.add_argument(
        '--save-debug-files',
        help='save camera images and HFR vs focuser position data for debugging or test',
        action='store_true',
    )
    parser.add_argument(
        '--plot',
        help='show plots of HFR vs focuser position',
        action='store_true',
    )
    cameras.add_program_arguments(parser)
    focusers.add_program_arguments(parser)
    logs.add_program_arguments(parser)
    targets.add_program_arguments(
        parser,
        allowed_types=TargetType.STAR | TargetType.EQUATORIAL,
        allow_sensor_fusion=False,
    )
    args = parser.parse_args()

    logs.setup_logging_from_args(args, 'autofocus')

    target_args = [args.target_type]
    if args.target_type == 'star':
        target_args.append(args.star)
        star_name = args.star.capitalize()
    elif args.target_type == 'coord-eq':
        # str() of a float is guaranteed to have enough precision such that when converted back to
        # a float it will be exactly the same as the original float.
        target_args.append(str(args.ra))
        target_args.append(str(args.dec))
        star_name = 'the bright star'

    camera = cameras.make_camera_from_args(args)
    focuser = focusers.make_focuser_from_args(args)
    focuser.motor_speed = 2  # fastest speed

    # Run the `track` program to get the mount pointed at the approximate location of the star.
    # Need to wait until the star is within the main camera's field of view before proceeding.
    logger.info(f'Moving mount until centered on {star_name}.')
    subprocess.run(
        args=[
            'track',
            '--no-console-logs',  # confusing to have logs from multiple processes
            '--fuse',
            f'--meridian-side={args.meridian_side}',
            '--stop-when-converged-angle=0.1',  # stop when within this many degrees of target
        ] + target_args,
        check=True,
    )

    # Run the `track` program as a subprocess in the background to continue tracking the star
    logger.info(f'Continuing to track {star_name}.')
    # pylint: disable=consider-using-with
    track_process = subprocess.Popen(
        args=[
            'track',
            '--no-console-logs',  # confusing to have logs from multiple processes
            f'--meridian-side={args.meridian_side}',
            '--fuse',
        ] + target_args
    )
    atexit.register(terminate_subprocess, track_process)
    time.sleep(5)  # mount will jump a bit when track process restarts

    if args.save_debug_files:
        # directory in which to save images for debugging purposes
        output_dir = os.path.join(
            DATA_PATH,
            'autofocus_' + datetime.utcnow().isoformat(timespec='seconds').replace(':', '')
        )
        os.makedirs(DATA_PATH, exist_ok=True)
        os.mkdir(output_dir)
        print(f'Saving debug files to {output_dir}')
    else:
        output_dir = None

    time_start = time.perf_counter()

    # Start by sparsely sampling the entire range to get a rough idea of where best focus is
    positions = np.linspace(focuser.min_position, focuser.max_position, 5, dtype=int)
    prelim_ideal_position = autofocus(
        camera,
        focuser,
        positions,
        skip_final_move=True,
        output_dir=output_dir,
        show_plot=args.plot,
    )

    # Refine by taking a greater number of closely spaced samples centered on the first estimate
    positions = np.linspace(
        start=max(prelim_ideal_position - 200, focuser.min_position),
        stop=min(prelim_ideal_position + 200, focuser.max_position),
        num=20,
        dtype=int)
    autofocus(camera, focuser, positions, output_dir=output_dir, show_plot=args.plot)

    if (returncode := track_process.poll()) is not None:
        logger.error(f'track subprocess ended prematurely with exit code {returncode}')

    time_elapsed = time.perf_counter() - time_start
    logger.info(f'Elapsed time: {time_elapsed} s')


if __name__ == "__main__":
    main()
