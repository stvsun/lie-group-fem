"""Dense chart-space L2 recovery and small P1 tetrahedron helpers."""

from dataclasses import dataclass

import numpy as np

from .groups import SO3Chart, _coordinates, _real, _shape


@dataclass(frozen=True, eq=False)
class RecoveredField:
    """A nodal coordinate field; retain its chart when evaluating new points."""

    chart: SO3Chart
    coordinates: np.ndarray
    weighted_rms: float
    stationarity_residual: float
    design_condition: float

    def __post_init__(self):
        coordinates = np.array(self.coordinates, dtype=np.float64, copy=True)
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def nodal_values(self):
        return self.chart.decode(self.coordinates)

    def evaluate(self, shape):
        """Evaluate target FE basis rows without re-encoding the nodal matrices."""
        return self.chart.decode(_shape(shape, len(self.coordinates)) @ self.coordinates)


def recover(values, shape, measures, chart):
    """Minimize sum_q measures[q] * ||(N @ C)[q] - chart(values[q])||^2.

    N has shape (n_quadrature, n_nodes); measures include physical quadrature
    weights and Jacobians. Solve the weighted design by SVD least squares,
    avoiding the squared condition number of explicitly solved mass equations.
    Positive measures and full column rank are required. No regularization,
    mass lumping, equilibrium enforcement, or material-interface averaging is
    applied implicitly. Partition material phases before calling this function.
    """
    samples = chart.encode(values)
    shape = _real(shape, "shape functions")
    if shape.ndim != 2 or shape.shape[1] == 0:
        raise ValueError("shape functions must be a nonempty matrix")
    shape = _shape(shape, shape.shape[1])
    measures = _real(measures, "measures")
    if shape.shape[0] != len(samples) or measures.shape != (len(samples),):
        raise ValueError("values, shape rows, and measures must have matching lengths")
    if np.any(measures <= 0):
        raise ValueError("quadrature measures must be positive")
    # Normalize before solving: a uniform change of length/volume units must
    # not affect the projection. Relative normalization also prevents overflow.
    weights = measures / np.max(measures)
    weights /= np.sum(weights)
    if np.any(weights == 0):
        raise ValueError("quadrature measures have unrepresentable dynamic range")
    design = np.sqrt(weights)[:, None] * shape
    rhs = np.sqrt(weights)[:, None] * samples
    coefficients, _, rank, singular = np.linalg.lstsq(design, rhs, rcond=None)
    if rank != shape.shape[1]:
        raise ValueError("rank-deficient recovery: add quadrature support or reduce the basis")
    condition = singular[0] / singular[-1]
    if condition >= 1e12:
        raise ValueError("recovery design is too ill-conditioned")
    chart.decode(coefficients)  # Fail early if recovered nodes leave this chart.
    residual = design @ coefficients - rhs
    return RecoveredField(
        chart=chart,
        coordinates=coefficients,
        weighted_rms=float(np.linalg.norm(residual)),
        stationarity_residual=float(np.linalg.norm(design.T @ residual)),
        design_condition=float(condition),
    )


def _tetrahedron(vertices):
    vertices = _coordinates(vertices, 3)
    if vertices.shape != (4, 3):
        raise ValueError("tetrahedron vertices must have shape (4, 3)")
    edges = (vertices[1:] - vertices[0]).T
    determinant = np.linalg.det(edges)
    if not np.isfinite(determinant) or determinant <= 0 or np.linalg.cond(edges) >= 1e12:
        raise ValueError("tetrahedron must have positive orientation and be nondegenerate")
    return vertices, edges, determinant


def tetrahedron_shape(vertices, points):
    """P1 barycentric rows at physical points in a positively oriented tet.

    Outside points are rejected: this helper never silently extrapolates.
    Coordinates and vertices must use the same length unit.
    """
    vertices, edges, _ = _tetrahedron(vertices)
    points = _coordinates(points, 3)
    local = np.linalg.solve(edges, (points - vertices[0]).T).T
    shape = np.column_stack([1 - local.sum(axis=1), local])
    if np.any(shape < -1e-12) or np.any(shape > 1 + 1e-12):
        raise ValueError("point lies outside the source tetrahedron")
    return shape


def tetrahedron_quadrature(vertices):
    """Four-point degree-two exact rule: physical points, positive volumes."""
    vertices, _, determinant = _tetrahedron(vertices)
    low = (5 - np.sqrt(5)) / 20
    high = (5 + 3 * np.sqrt(5)) / 20
    shape = np.full((4, 4), low)
    np.fill_diagonal(shape, high)
    return shape @ vertices, np.full(4, determinant / 24)
