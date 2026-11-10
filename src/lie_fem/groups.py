"""Local SO(3) and polar GL+(3) charts; all arrays use float64.

Matrices act on column vectors. The reference is a proper rotation, not a
deformation gradient. The open rotation ball excludes the principal-log cut.
GL3Chart means the orientation-preserving component only.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

_PI_MARGIN = 1e-7
_ROT_TOL = 1e-10
_MAX_CONDITION = 1e12
_SQRT2 = np.sqrt(2.0)


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _matrices(value):
    array = _real(value, "matrices")
    if array.ndim != 3 or array.shape[1:] != (3, 3) or len(array) == 0:
        raise ValueError("matrices must have shape (n, 3, 3), n > 0")
    return array


def _rotations(value):
    array = _matrices(value)
    error = np.linalg.norm(array.transpose(0, 2, 1) @ array - np.eye(3), axis=(1, 2))
    if np.any(error > _ROT_TOL) or np.any(np.abs(np.linalg.det(array) - 1) > _ROT_TOL):
        raise ValueError("expected proper orthogonal rotation matrices")
    return array


def _coordinates(value, dimension):
    array = _real(value, "coordinates")
    if array.ndim != 2 or array.shape[1] != dimension or len(array) == 0:
        raise ValueError(f"coordinates must have shape (n, {dimension}), n > 0")
    return array


def _inside_chart(vectors):
    if np.any(np.linalg.norm(vectors, axis=1) >= np.pi - _PI_MARGIN):
        raise ValueError("rotation reaches the pi branch cut; choose another reference or patch")


def _shape(value, n_nodes):
    array = _real(value, "shape functions")
    if array.ndim != 2 or array.shape[1] != n_nodes or len(array) == 0:
        raise ValueError(f"shape functions must have shape (n_points, {n_nodes})")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("shape functions must form a partition of unity")
    return array


@dataclass(frozen=True, eq=False)
class SO3Chart:
    """R = reference @ exp(hat(r)); r in the open principal rotation ball.

    Supply a common reference near all samples. Changing it changes the chart
    interpolation for noncommuting rotations. No automatic branch unwrapping.
    """

    reference: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        reference = _real(self.reference, "reference")
        if reference.shape != (3, 3):
            raise ValueError("reference must have shape (3, 3)")
        reference = _rotations(reference[None])[0].copy()
        reference.setflags(write=False)
        object.__setattr__(self, "reference", reference)

    def encode(self, rotations):
        """Map (n,3,3) rotations to (n,3) rotation vectors, in radians."""
        relative = self.reference.T @ _rotations(rotations)
        vectors = Rotation.from_matrix(relative).as_rotvec()
        _inside_chart(vectors)
        return vectors

    def decode(self, coordinates):
        """Map (n,3) chart vectors to (n,3,3) rotations."""
        vectors = _coordinates(coordinates, 3)
        _inside_chart(vectors)
        return self.reference @ Rotation.from_rotvec(vectors).as_matrix()


@dataclass(frozen=True, eq=False)
class GL3Chart(SO3Chart):
    """F = R U in GL+(3), encoded by rotation vector and symmetric log(U).

    Nine coordinates: r_x, r_y, r_z, L_xx, L_yy, L_zz,
    sqrt(2)*L_yz, sqrt(2)*L_xz, sqrt(2)*L_xy. The last six preserve the
    Frobenius inner product of symmetric matrices. SVD vectors are used only
    to reconstruct polar factors; individual eigenframes are never mapped.
    """

    def encode(self, gradients):
        gradients = _matrices(gradients)
        if np.any(np.linalg.slogdet(gradients)[0] <= 0):
            raise ValueError("GL3Chart requires det(F) > 0")
        left, singular, right = np.linalg.svd(gradients)
        if np.any(singular[:, -1] <= singular[:, 0] / _MAX_CONDITION):
            raise ValueError("deformation gradient is singular or too ill-conditioned")
        rotations = left @ right
        vectors = SO3Chart.encode(self, rotations)
        logarithm = (right.transpose(0, 2, 1) * np.log(singular)[:, None, :]) @ right
        packed = np.column_stack([
            logarithm[:, 0, 0], logarithm[:, 1, 1], logarithm[:, 2, 2],
            _SQRT2 * logarithm[:, 1, 2], _SQRT2 * logarithm[:, 0, 2],
            _SQRT2 * logarithm[:, 0, 1],
        ])
        return np.column_stack([vectors, packed])

    def decode(self, coordinates):
        coordinates = _coordinates(coordinates, 9)
        rotations = SO3Chart.decode(self, coordinates[:, :3])
        packed = coordinates[:, 3:]
        logarithm = np.zeros((len(packed), 3, 3), dtype=np.float64)
        logarithm[:, range(3), range(3)] = packed[:, :3]
        for column, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
            logarithm[:, i, j] = logarithm[:, j, i] = packed[:, column] / _SQRT2
        eigenvalues, vectors = np.linalg.eigh(logarithm)
        if np.any(eigenvalues[:, -1] - eigenvalues[:, 0] >= np.log(_MAX_CONDITION)):
            raise ValueError("reconstructed stretch is too ill-conditioned")
        try:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                stretches = (vectors * np.exp(eigenvalues)[:, None, :]) @ vectors.transpose(0, 2, 1)
                gradients = rotations @ stretches
        except FloatingPointError as error:
            raise ValueError("stretch is outside the representable float64 range") from error
        if not np.all(np.isfinite(gradients)) or np.any(np.linalg.slogdet(gradients)[0] <= 0):
            raise ValueError("reconstructed deformation is not numerically in GL+(3)")
        return gradients


def interpolate(values, shape, chart):
    """Interpolate samples using rows of FE weights in one explicit chart.

    Signed weights are allowed (e.g. higher-order elements), but the result
    must stay inside the chart and numerical conditioning bounds.
    """
    coordinates = chart.encode(values)
    return chart.decode(_shape(shape, len(coordinates)) @ coordinates)
