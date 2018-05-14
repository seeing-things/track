"""A collection of assorted math operations."""

import math
import numpy as np


def wrap_error(error):
    """Wraps an angle in degrees to the range [-180,+180)"""
    return (error + 180.0) % 360.0 - 180.0

def clamp(val, limit):
    """Limit value to symmetric range.

    Args:
        val: Value to be adjusted.
        limit: Absolute value of return value will not be greater than this.

    Returns:
        The input value limited to the range [-limit,+limit].
    """
    return max(min(limit, val), -limit)

def normalize(v):
    """Normalize a Cartesian vector in 3-space to unit magnitude.

    Args:
        v: A vector in 3-space using Cartesian coordinates.

    Returns:
        A vector in 3-space with the same direction as v but with unit magnitude.
    """
    v = np.asarray(v)
    return v / math.sqrt(np.dot(v, v))

def rotate(v, axis, theta):
    """Rotates a vector in 3-space.

    Rotate a vector v in 3-space about the given axis of rotation by theta in a counterclockwise
    direction (right-hand rule). The rotation matrix is formed using the Euler-Rodrigues formula.

    Reference: https://stackoverflow.com/a/6802723

    Args:
        v: A vector in 3-space using Cartesian coordinates.
        axis: A vector in 3-space giving the axis of rotation. Not required to be a unit vector
            (magnitude ignored).
        theta: Angle of rotation in degrees. The rotation will be performed according to the
            right-hand rule convention.

    Returns:
        A numpy array with size 3 representing the rotated vector.
    """
    # pylint: disable=too-many-locals

    theta = theta * math.pi / 180.0

    # make the rotation matrix
    axis = np.asarray(axis)
    axis = normalize(axis)
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return np.dot(rotation_matrix, v)

def horiz_to_cart(v):
    """Convert unit vector from horizontal to Cartesian coordinate system.

    Converts a unit vector in 3-space from the horizontal (Azimuth-Altitude) coordinate system to
    the Cartesian (x,y,z) coordinate system. The horizontal coordinate system is similar to the
    spherical coordinate system except (1) the azimuthal angle increases in a clockwise direction
    about the z-axis, and (2) the altitude is zero at the azimuthal (xy) plane, in contrast to the
    polar angle of the spherical coordinate system which is zero at zenith.

    Args:
        v: A dict containing keys 'az' and 'alt' with values in degrees. It is assumed that the
            magnitude of this vector is 1.

    Returns:
        A size 3 numpy array containing the Cartesian coordinates of the vector.
    """

    # degrees to radians
    az = v['az'] * math.pi / 180.0
    alt = v['alt'] * math.pi / 180.0

    # horizontal to spherical
    phi = -az
    theta = math.pi / 2.0 - alt

    # spherical to Cartesian
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return np.asarray([x, y, z], dtype=float)

def equatorial_to_cart(v):
    """Convert unit vector from equatorial to Cartesian coordinate system.

    Converts a unit vector in the equatorial coordinate system from a pair of right ascension (RA)
    and declination (DEC) coordinates to Cartesian (x,y,z) coordinates. The equatorial coordinate
    system is similar to the spherical coordinate system except the declination is zero at the
    equator, in contrast to the polar angle of the spherical coordinate system which is zero at
    the North pole.

    Args:
        v: A dict containing keys 'ra' and 'dec' with values in degrees. It is assumed that the
            magnitude of this vector is 1.

    Returns:
        A size 3 numpy array containing the Cartesian coordinates of the vector.
    """

    # degrees to radians
    ra = v['ra'] * math.pi / 180.0
    dec = v['dec'] * math.pi / 180.0

    # equatorial to spherical
    phi = ra
    theta = math.pi / 2.0 - dec

    # spherical to Cartesian
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return np.asarray([x, y, z], dtype=float)

def cart_to_horiz(v):
    """Convert a vector from Cartesian to horizontal coordinate system.

    Converts a vector in 3-space from the Cartesian (x,y,z) coordinate system to the horizontal
    (Azimuth-Altitude) coordinate system. Any magnitude information will be lost. See the
    horiz_to_cart description for differences between horizontal and spherical coordinate systems.

    Args:
        v: A size 3 numpy array containing the Cartesian coordinates of a vector.

    Returns:
        A dict containing keys 'az' and 'alt' with values in degrees.
    """
    x = v[0]
    y = v[1]
    z = v[2]

    # Cartesian to spherical (unit magnitude assumed)
    theta = math.acos(z)
    phi = math.atan2(y, x)

    # spherical to horizontal
    az = (-phi) % (2.0*math.pi)
    alt = math.pi / 2.0 - theta

    # radians to degrees
    az_deg = 180.0 / math.pi * az
    alt_deg = 180.0 / math.pi * alt

    return {'az': az_deg, 'alt': alt_deg}

def cart_to_equatorial(v):
    """Convert a vector from Cartesian to equatorial coordinate system.

    Converts a vector in 3-space from the Cartesian (x,y,z) coordinates to right ascension (RA) and
    declination (DEC) coordinates in the equatorial coordinate system. Any magnitude information
    will be lost. See the equatorial_to_cart description for differences between equatorial and
    spherical coordinate systems.

    Args:
        v: A size 3 numpy array containing the Cartesian coordinates of a vector.

    Returns:
        A dict containing keys 'ra' and 'dec' with values in degrees.
    """
    x = v[0]
    y = v[1]
    z = v[2]

    # Cartesian to spherical (unit magnitude assumed)
    theta = math.acos(z)
    phi = math.atan2(y, x)

    # spherical to equatorial
    ra = phi % (2.0*math.pi)
    dec = math.pi / 2.0 - theta

    # radians to degrees
    ra_deg = 180.0 / math.pi * ra
    dec_deg = 180.0 / math.pi * dec

    return {'ra': ra_deg, 'dec': dec_deg}

def angle_between(u, v):
    """Angle between two vectors.

    Compute the angle between two vectors. The vectors can be of any dimension though in the
    context of this module 3-space is anticipated to be the likely use-case. The algorithm for
    computing the angle was taken from the following paper which claims that it has better
    numerical performance than other more common approaches (see page 47):
    https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf

    Args:
        u, v: Input vectors. If Cartesian coordinates these should both be either lists or numpy
            arrays. If horizontal coordinates these should both be dicts containing keys 'az' and
            'alt'.

    Returns:
        Angle between the vectors in degrees.
    """

    if isinstance(u, dict) and isinstance(v, dict):
        if set(['az', 'alt']).issubset(set(u.keys())):
            u = horiz_to_cart(u)
            v = horiz_to_cart(v)
        elif set(['ra', 'dec']).issubset(set(u.keys())):
            u = equatorial_to_cart(u)
            v = equatorial_to_cart(v)
        else:
            raise ValueError("dict keys must be ['az','alt'] or ['ra','dec']")
    else:
        u = np.asarray(u, dtype='float')
        v = np.asarray(v, dtype='float')

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    theta = 2.0 * np.arctan2(np.linalg.norm(v_norm * u - u_norm * v),
                             np.linalg.norm(v_norm * u + u_norm * v))
    return theta * 180.0 / math.pi

def adjust_position(target_position_prev, target_position, offset):
    """Adjust target position by correction factor.

    Adjusts the position of the target by an offset where the offset is specified in a reference
    frame defined by the object's direction of travel. This makes it possible to make adjustments
    such as "1 degree ahead of the predicted position" or "0.3 degrees left with respect to the
    object's direction of travel."  This is expected to be more useful than adjustments such as
    "1 degree higher in altitude."

    Args:
        target_position_prev: A dict with keys for each axis giving the position of the target a
            short time ago in degrees.
        target_position: A dict with keys for each axis giving the current position of the target
            in degrees.
        offset: A two-element list or numpy array giving the offset adjustments in degrees. The
            first element is the x-axis offset and the second element is the y-axis offset. The +y
            axis is in the direction of the object's motion. The x-axis is perpendicular to the
            object's motion.

    Returns:
        A dict with keys for each axis giving the adjusted position.
    """
    if set(target_position.keys()) == set(['az', 'alt']):
        coord_system = 'horizontal'
    elif set(target_position.keys()) == set(['ra', 'dec']):
        coord_system = 'equatorial'
    else:
        raise ValueError("dict keys must be ['az','alt'] or ['ra','dec']")

    # convert vectors in spherical coordinate systems to Cartesian
    if coord_system == 'horizontal':
        tpos_prev = horiz_to_cart(target_position_prev)
        tpos = horiz_to_cart(target_position)
    else:
        tpos_prev = equatorial_to_cart(target_position_prev)
        tpos = equatorial_to_cart(target_position)

    if all(tpos == tpos_prev):
        raise ValueError('current and previous positions are equal!')

    # Compute object motion vector. This makes the assumption that tpos and tpos_prev are separated
    # by a small angle. The more correct calculation would result in a vector tmotion that is
    # tangent to the unit sphere at the location of tpos.
    tmotion = normalize(tpos - tpos_prev)

    # convert gamepad vector from Cartesian to polar
    gpcomplex = offset[0] + 1j*offset[1]
    gpmag = np.abs(gpcomplex)
    gparg = np.angle(gpcomplex) * 180.0 / math.pi

    # compute axis of rotation
    axis = rotate(tmotion, tpos, gparg)

    # rotate position about axis of rotation
    tpos_new = rotate(tpos, axis, gpmag)

    # convert back to original coordinate system
    if coord_system == 'horizontal':
        return cart_to_horiz(tpos_new)
    else:
        return cart_to_equatorial(tpos_new)

def camera_eq_error(mount_position, target_xy, degrees_per_pixel):
    """Compute camera target error components in equatorial coordinate system.

    When a camera is used to detect a target the location of the target is easily found in the 2-D
    grid of the frame. This must somehow be transformed into error components for the two axes of
    the mount which exist in a spherical coordinate system. This function performs that transform.
    Unfortunately the mount position must be known to do a perfect transformation. However the
    accuracy of the mount position need not be highly accurate. What is most important is that 90
    degrees declination should correspond fairly closely to the point where the declination axis
    actually crosses the meridian. The error terms are most sensitive when the mount is pointing
    near the pole since it is a mathematical singularity.

    Args:
        mount_position: Dict with keys 'ra' and 'dec' giving the position of the mount in the
            equatorial coordinate system in degrees.
        target_xy: Position of target in the camera's frame in pixels where (0, 0) is the center
            of the frame. The Y-axis of the camera frame is assumed to be parallel to the
            declination axis.
        degrees_per_pixel: Apparent size of a photosite in degrees.

    Returns:
        A dict with keys 'ra' and 'dec' giving the error components of the target relative to
            the center of the camera frame in degrees.
    """

    z_axis = np.array([0, 0, 1])
    x_axis = np.array([1, 0, 0])


    # get equatorial coordinates of target

    # start with cartesian coordinates of target assuming camera center is pointed at the
    # celestial pole, normalized to unit magnitude
    focal_length_px = 1.0 / np.tan(np.deg2rad(degrees_per_pixel))
    target_cart = np.array([target_xy[0], target_xy[1], -focal_length_px])
    target_cart = target_cart / np.linalg.norm(target_cart)

    # rotate to account for where the mount is pointed in the equatorial coordinate system
    target_cart = rotate(target_cart, x_axis, mount_position['dec'] + 90.0)
    target_cart = rotate(target_cart, z_axis, mount_position['ra'] - 90.0)

    # convert from cartesian to equatorial
    target_eq = cart_to_equatorial(target_cart)


    # determine error components
    error = {
        'ra': wrap_error(target_eq['ra'] - mount_position['ra']),
        'dec': target_eq['dec'] - mount_position['dec']
    }

    # RA reversal depending on side of meridian
    if mount_position['pdec'] > 180.0:
        error['ra'] = -error['ra']

    return error