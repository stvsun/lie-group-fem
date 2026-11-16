"""Affine tensor-product Q1 boxes for the analytic 2013 paper benchmarks.

This study helper is intentionally limited to structured rectangular grids.
It neither reads arbitrary meshes nor runs a mechanics simulation.
"""

from itertools import product
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

BITS = np.array(list(product((0, 1), repeat=3)))


def shape(local):
    return np.prod(np.where(BITS[None] == 1, local[:, None], 1 - local[:, None]), axis=2)


class BoxMesh:
    def __init__(self, counts, lower, upper):
        self.counts = np.array(counts, dtype=int)
        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)
        if np.any(self.counts < 1) or np.any(self.upper <= self.lower):
            raise ValueError("positive counts and box lengths required")
        self.spacing = (self.upper - self.lower) / self.counts
        self.nodes = np.array(list(product(*[
            np.linspace(a, b, n + 1) for a, b, n in zip(self.lower, self.upper, self.counts)])))
        origins = np.array(list(product(*[range(n) for n in self.counts])))
        self.origins = self.lower + origins * self.spacing
        self.cells = np.ravel_multi_index((origins[:, None] + BITS).transpose(2, 0, 1), self.counts + 1)

    def quadrature(self, order=2):
        abscissae, weights = leggauss(order)
        local = np.array(list(product((abscissae + 1) / 2, repeat=3)))
        tensor_weights = np.prod(np.array(list(product(weights / 2, repeat=3))), axis=1)
        points = (self.origins[:, None] + local * self.spacing).reshape(-1, 3)
        measures = np.tile(tensor_weights * np.prod(self.spacing), len(self.cells))
        return points, measures, self.basis(points)

    def basis(self, points):
        points = np.asarray(points)
        scaled = (points - self.lower) / self.spacing
        if np.any(scaled < -1e-10) or np.any(scaled > self.counts + 1e-10):
            raise ValueError("point lies outside source box")
        indices = np.minimum(np.floor(np.maximum(scaled, 0)).astype(int), self.counts - 1)
        local = scaled - indices
        cells = np.ravel_multi_index(indices.T, self.counts)
        return coo_matrix((shape(local).ravel(),
                           (np.repeat(np.arange(len(points)), 8), self.cells[cells].ravel())),
                          shape=(len(points), len(self.nodes))).tocsr()


def project(basis, weights, values):
    """Consistent mass solve; all paper grids have full quadrature support."""
    weights = weights / weights.sum()
    weighted = basis.multiply(weights[:, None])
    mass = (basis.T @ weighted).tocsc()
    rhs = basis.T @ (weights[:, None] * values.reshape(len(values), -1))
    coefficients = splu(mass).solve(rhs)
    residual = np.linalg.norm(mass @ coefficients - rhs)
    if residual > 1e-11 * max(1., np.linalg.norm(rhs)):
        raise ArithmeticError("mass solve failed its stationarity check")
    return coefficients, float(residual)


def rotation(theta):
    result = np.tile(np.eye(3), (len(theta), 1, 1))
    result[:, 0, 0] = result[:, 1, 1] = np.cos(theta)
    result[:, 1, 0] = np.sin(theta)
    result[:, 0, 1] = -np.sin(theta)
    return result


def ring_fields(points):
    """Eqs. 6.2--6.4 with L=1; analytic continuous rotation lift Eq. 6.7."""
    theta = 2 * np.pi * points[:, 0]
    stretch = np.tile(np.eye(3), (len(points), 1, 1))
    stretch[:, 0, 0] = 1 - 2 * np.pi * points[:, 1]
    rotations = rotation(theta)
    return rotations @ stretch, rotations, stretch


def ring_coordinates(points, method):
    f, r, s = ring_fields(points)
    if method == "direct_F":
        return f.reshape(-1, 9)
    if method == "direct":
        return np.column_stack([r.reshape(-1, 9), s[:, 0, 0]])
    return np.column_stack([2 * np.pi * points[:, 0],
                            np.log(s[:, 0, 0]) if method == "lie" else s[:, 0, 0]])


def ring_decode(coordinates, method):
    if method == "direct_F":
        return coordinates.reshape(-1, 3, 3), None, None
    if method == "direct":
        r = coordinates[:, :9].reshape(-1, 3, 3)
    else:
        r = rotation(coordinates[:, 0])
    s = np.tile(np.eye(3), (len(coordinates), 1, 1))
    s[:, 0, 0] = np.exp(coordinates[:, -1]) if method == "lie" else coordinates[:, -1]
    return r @ s, r, s


def error(approximate, exact, weights):
    squared = np.sum(np.abs(approximate - exact) ** 2, axis=(1, 2))
    return float(np.sqrt(np.dot(weights, squared)))


def recovered_ring(mesh, method):
    points, weights, basis = mesh.quadrature(2)
    if method not in ("nearest", "extrapolation"):
        coefficients, residual = project(basis, weights, ring_coordinates(points, method))
        return coefficients, residual, method
    samples = ring_fields(points)[0].reshape(len(mesh.cells), 8, 9)
    if method == "nearest":
        local_values = samples
    else:
        g = (np.array([-1., 1.]) / np.sqrt(3) + 1) / 2
        gauss_basis = shape(np.array(list(product(g, repeat=3))))
        local_values = np.linalg.solve(gauss_basis, samples)
    coefficients = np.zeros((len(mesh.nodes), 9))
    count = np.zeros(len(mesh.nodes))
    np.add.at(coefficients, mesh.cells.ravel(), local_values.reshape(-1, 9))
    np.add.at(count, mesh.cells.ravel(), 1)
    return coefficients / count[:, None], 0., "direct_F"
