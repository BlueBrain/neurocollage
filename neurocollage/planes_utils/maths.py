"""Maths utilities for plane creation.

The functions of this module are functions acting on oriented affine planes of the 3D Euclidean
space. These functions are used by both the plane module and the coordinates module.

A Plane object as defined in this module is the data consisting of a distinguihed 3D point [x, y, z]
and a unit vector [A, B, C]. The underlying oriented affine plane is the plane passing
through [x, y, z] and orthogonal to the line defined by [A, B, C]. An equation defining the plane
is given by A * X + B * Y + C * Z = D where D = A * x + B * y + C * z.

The 'anchor' point [x, y, z] can be used for rendering purpose.


Output formats
**************

When saved to disk, the planes created by the planes module can be represented under two different
 output formats:

- the so-called quaterionic format: [x, y, z, a, b, c, d]
- the point-normal or Plane format: [x, y, z, A, B, C]

The quaternionic format is a legacy output format. It is maintained for backward compatibility.

For both output formats the float vector (x, y, z) represents the 3D coordinates of a
distinguished point of the plane P. This point is an anchor that can be used as the origin
of a 2D orthonormal frame of plane for visualization purposes. Usually, the point (x, y, z) is
the intersection point of P with a computed centerline.

With the format [x, y, z, a, b, c, d] of a plane P, the [a, b, c, d]-part is a
unit quaternion q complying with the convention 'w, x, y, z'. The quaternion q =
w + x * i + y * j + z * k maps ZVECTOR = [0, 0, 1] to a unit vector orthogonal to P,
 i.e., qkq^{-1} = n_x * i + n_y * j + n_z * k where (n_x, n_y, n_z) is a normal unit vector of P.
 See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.

With the format [x, y, z, A, B, C] of a plane P, the float vector [A, B, C] is the unit normal
vector of the plane.
"""

import warnings

import numpy as np
from pyquaternion import Quaternion

# pylint: disable=broad-exception-raised
XVECTOR = np.array([1, 0, 0])
YVECTOR = np.array([0, 1, 0])
ZVECTOR = np.array([0, 0, 1])
CANONICAL = np.array([XVECTOR, YVECTOR, ZVECTOR])
ZVECTOR = np.array([0, 0, 1])


def normalize_vectors(vectors):
    """Normalize a bunch of vectors along axis==1.

    Args:
        vectors: vectors with shape
         [[x1, y1, z1],
         [x2, y2, z2],
         [x3, y3, z3],
         ...
        ]
    """
    return vectors / np.sqrt(np.einsum("...i,...i", vectors, vectors)).reshape(-1, 1)


def normalize_vector(vector):
    """Return a normalize vector (a bit faster for one vector than normalize_vectors)."""
    return vector / np.linalg.norm(vector)


class Plane:
    """Class to perform algebro-geometric operations on planes defined in 3D space.

    The underlying abstract data is an affine hyperplane of the 3D Euclidean space together with a
    distinguished point lying in this plane. (Hence a Plane is a 'pointed' 2D plane in 3D.)

    Attributes:
        point: numpy.ndarray of shape (3,) representing a distinguished point [x, y, z] of the
            plane which can be used as the origin of an orthonormal basis of the plane.
        normal: numpy.ndarray of shape (3,) holding a unit normal vector [A, B, C]
            The planes is defined as the locus of the points [X, Y, Z] such that
            A * X + B * Y + C * Z = D where D = A * x + B * y + C * z.
    """

    def __init__(self, point, normal):
        self.point = np.array(point)
        self.normal = np.array(normal)

        if len(point) != 3:
            raise Exception(f"'point' is of length {len(point)}. Expected length: 3")

        if len(normal) != 3:
            raise Exception(f"'normal' is of length {len(point)}. Expected length: 3")

        self.normalize()

    def normalize(self):
        """Normalize the orthogonal vector used to define the plane.

        The vector self.normal = [A, B, C] holding the first 3 coefficients of the plane equation is
        normalized.

        Raises:
            Exception if the normal vector `normal` is zero.
        """
        norm = np.linalg.norm(self.normal)
        if norm == 0.0:
            raise Exception("Zero normal vector [A, B, C]: Cannot define a plane.")
        self.normal = self.normal / norm

    def get_quaternion(self, reference_vector=ZVECTOR):
        """Returns a quaternion which maps `reference_vector` to the equation's unit normal.

        Args:
            reference_vector: (Optional) array [x, y, z] representing a 3D vector that is mapped to
            to the plane normal vector [A, B, C] by the output quaternion.
            coefficients. Defaults to ZVECTOR=[0, 0, 1].

        Returns:
            A unit Quaternion([w, x, y, z]) which maps `reference_vector` to the plane unit normal
            vector [A, B, C].
        """
        reference_vector = normalize_vector(reference_vector)
        self.normalize()

        return Quaternion(quaternion_from_vectors(reference_vector, self.normal)).unit

    def get_basis(self, reference_basis=CANONICAL):
        """Returns a direct orthonormal basis of the plane.

        The returned basis [u, v] is such that [u, v, [A, B, C]] is a direct orthonormal basis
        of R^3 where [A, B, C] is the plane unit normal vector.

        The construction is as follows:

        - build a quaternion q that maps the 3D vector W of the reference basis [U, V, W] to
         to the unit normal vector [A, B, C] of the plane (see quaternion_from_vectors).
        - rotate [U, V] by q to get the output basis [u, v].

        Args:
            reference_basis: np.ndarray array [U, V, W] of shape (3, 3) representing a direct
                orthonormal basis of R^3. The pair [U, V] is a direct orthonormal basis of the
                reference plane defined by the equation W dot [X, Y, Z] = 0.

        Returns:
            numpy.ndarray of shape (2, 3) holding the two vectors of a direct orthonormal basis of
            the plane.
        """

        def is_direct_orthonormal_basis(basis):
            w = np.cross(basis[0], basis[1])
            norms = np.linalg.norm(basis, axis=1)
            diff_vec = np.linalg.norm(basis[2] - w)
            diff_norms = np.linalg.norm(norms - np.array([1.0, 1.0, 1.0]))

            return np.isclose(diff_norms, 0.0) and np.isclose(diff_vec, 0.0)

        if not is_direct_orthonormal_basis(reference_basis):
            raise Exception(
                f"The reference basis {reference_basis}"
                " is not a direct orthonormal basis of R^3."
            )

        quaternion = self.get_quaternion(reference_basis[2])
        return np.array([quaternion.rotate(reference_basis[i]) for i in range(2)])

    def get_best_alignment_basis(self, target, axis=0):
        """Returns a direct orthonormal basis of the plane.

        Depending on the value of `axis`, either the first or the second basis vector is chosen
        so that it has the largest possible scalar product with `target` (i.e., as close as
        possible to 1.0 when the target vector is normalized).

        If `target` is colinear with the plane unit normal vector [A, B, C], the basis is not
        unique: any unit vector orthogonal to [A, B, C] has the best alignment with `target`.
        In this case the function returns self.get_basis().

        Args:
            target: array-like object of shape (3,); this is the 3D vector with which the best
                alignment is requested.
            axis(int): (optional) either 0 or 1, depending on whether the first or the second basis
                vector is requested to be as close as possible to `target` wrt to the 3D Euclidean
                norm (this is equivalent to scalar product maximisation). Defaults to 0, i.e., the
                first basis vector is best aligned with `target`.

        Returns:
            np.ndarray of shape (2, 3) holding the two vectors of a direct orthonormal basis of
            the plane. Depending on the value of `axis`, either the first or the second basis vector
            has the best possible alignment with `target` among all possible basis vectors.

        Raises:
            Exception if `target`is close to zero, see np.close documentation,
            https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

        """
        if axis not in [0, 1]:
            raise Exception(f'"axis" must be either 0 or 1. Got {axis}')

        norm = np.linalg.norm(target)
        if np.isclose(norm, 0.0):
            raise Exception(
                'The "target" vector is close to zero. A non-zero 3D vector is expected.'
            )

        basis = np.zeros((2, 3), dtype=float)
        basis[1 - axis] = np.cross(target, self.normal)
        norm = np.linalg.norm(basis[1 - axis])
        if np.isclose(norm, 0.0):
            warnings.warn(
                f"The target vector {target} is colinear with the plane normal {self.normal}"
                "Returning the default basis of Plane.get_basis.",
                UserWarning,
            )
            return self.get_basis()

        basis[1 - axis] /= norm
        basis[axis] = np.cross(self.normal, basis[1 - axis])
        if axis == 0:
            basis[1 - axis] *= -1.0

        return basis

    @classmethod
    def from_quaternion(cls, point, quaternion, reference_vector=ZVECTOR):
        """Instantiate a plane from a quaternion and an associated `reference_vector`.

        Args:
            point: list or numpy.ndarray [x, y, z] of shape (3,) representing a distinguished
                point of the plane to be created.
            quaternion: a quaternion q complying with the [w, x ,y, z] convention, i.e.,
                q = w + x * i + y * j + z * k. `quaternion` is any value supported by the
                constructor of pyquaternion.Quaternion. It can be for instance a 4D array
                [w, x, y, z] or a pyquaternion.Quaternion instance.
            reference_vector: (Optional) list or numpy.ndarray [n_x, n_y, n_z] of shape (3,) whose
                image by the rotation of `quaternion` defines a unit normal vector of the oriented
                plane to be created. Defaults to ZVECTOR=(0, 0, 1).

        Returns:
            A Plane object passing through `point` and whose unit normal vector [A, B, C] is
            obtained via rotation the `reference_vector` by `quaternion`.

        """
        try:
            quaternion = Quaternion(quaternion)
        except ValueError as error:
            raise Exception(
                f"Cannot convert 'quaternion' {quaternion} to pyquaternion.Quaternion"
            ) from error
        normal = quaternion.rotate(reference_vector)

        return cls(point, normal)

    def to_numpy(self):
        """Return an array of shape (7,) concatenating the anchor point and the unit normal vector.

        Returns:
          numpy.ndarray of shape (7,) of the form [x, y, z, A, B, C] where [x, y, z] is
            a disinguished point of the plane and [A, B, C] is the unit normal vector of the plane.
        """
        return np.concatenate([self.point, self.normal])

    def get_equation(self):
        """Return an array of shape (4,) containing the coefficients of a normalized plane equation.

        Returns:
          numpy.ndarray of shape (4,) of the form [A, B, C, D] where [A, B, C] is the unit normal
          vector of the plane and D = A * x + B * y + C * z, with point = [x, y, z] the anchor point
          of the plane.
        """
        self.normalize()
        return np.concatenate([self.normal, [np.dot(self.point, self.normal)]])


def create_orthogonal_planes(start, end, points):
    """Create a sequence of Planes orthogonal to the line segment [start, end].

    The function returns a list of Plane objects. Each underlying plane has a unit normal vector
    with direction end - start. The positions of the plane anchor points between `start` and `end`
    are determined by the 1D array `points`.

    Args:
        start: an array-like object of shape (3,). The start point of the 3D line segment
            to which the created planes are orthogonal.
        end: an array-like object of shape (3,). The end point of the 3D line segment
            to which the created planes are orthogonal.
        points: a 1D array-like object. An sequence of numbers representing the positions
             of the orthogonal planes along the segment [start, end].
             Example: If `points` is [0, 0.5, 1.0], then 3 planes orthogonal to [start, end] will
             be created. The first one passes through `start`, the second one passes through
             `0.5 * (start + point)` and the last one passes through `end`.

    Returns:
        list of Plane objects of the same length as `points`. The returned Plane objects have
        all the same unit normal vector, that is `(end - start) / np.linal.norm(end - start)`.

    Raises: Exception if:
        * `start` and `end` are close to be the same (see numpy.close documentation)
        * `points` is not a 1D array-like object

    """
    diff = np.asarray(end) - np.asarray(start)
    norm = np.linalg.norm(diff)
    if np.isclose(norm, 0.0):
        raise Exception(
            'The "end" point is close to the "start" point. Two distinct 3D points are expected.'
        )
    normal = diff / norm  # the common plane unit normal vector

    points = np.asarray(points)
    if len(points.shape) > 1:
        raise Exception(f'"points" must be a 1D array. Got shape {points.shape}')

    # Affine transform of `points` into [0, 1] so that min(points) = 0 and max(points) = 1.
    points -= np.min(points)
    points = points / np.max(points)

    # A sequence of anchor points along the 3D line segment [start, end]
    anchors = np.outer(points, diff) + np.array(start)

    return [Plane(anchor, normal) for anchor in anchors]


def distances_to_planes(point, planes):
    """Returns the signed distance of a point to a list of planes.

    Args:
        point: the point [x, y, z] to compute the distances to.
        planes: the list of Plane objects.

    Returns: signed distance (np.array([d1, ..., d2])) to all planes.
    """
    normals = [plane.normal for plane in planes]
    point = np.asarray(point)
    points = [plane.point for plane in planes]
    return np.einsum("ij,ij->i", normals, point - points) / np.linalg.norm(normals, axis=1)


def quaternion_from_vectors(s, t):
    """Returns the quaternion (s cross t) + (s dot t + |s||t|).

    This quaternion q maps s to t * |s|/|t|, i.e., qsq^{-1} = t * |s|/|t|.
    Hence q maps s to t if s and t have the same norm.

    Note: if s or t is zero, the returned quaternion [w, x, y, z] is
    a zero-vector which cannot be interpreted as a direct isometry of R^3.

    Args:
        s(numpy.ndarray): numeric array of shape (3,).
        t(numpy.ndarray): numeric array of shape (3,).

    Returns:
        Numeric array of shape (4,).
        This data is interpreted as a quaternion. A quaternion is a 4D
        vector [w, x, y, z] where [x, y, z] is the imaginary part.
    """
    w = np.dot(s, t) + np.linalg.norm(s) * np.linalg.norm(t)
    return np.hstack([w, np.cross(s, t)])
