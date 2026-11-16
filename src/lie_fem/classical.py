"""Real local charts for all eight group families in Mota et al. Table 1.

MatrixChart uses A = reference @ exp(B), not the polar GL3Chart. The
principal logarithm must be real and in the declared algebra. This is a local
chart, not a global inverse of the exponential. Sp(n) means real 2n by 2n.
"""

import numpy as np
from scipy.linalg import expm, logm


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


class PositiveRealChart:
    """Multiplicative positive scalars; values and coordinates have shape (q,1)."""

    dimension = 1

    @staticmethod
    def encode(values):
        values = _real(values, "positive scalars")
        if values.ndim != 2 or values.shape[1] != 1 or np.any(values <= 0):
            raise ValueError("positive scalars must have shape (q,1) and be > 0")
        return np.log(values)

    @staticmethod
    def decode(coordinates):
        coordinates = _real(coordinates, "coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 1:
            raise ValueError("coordinates must have shape (q,1)")
        with np.errstate(over="ignore", under="ignore"):
            values = np.exp(coordinates)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("positive scalar exponential is unrepresentable")
        return values


def _basis(group, n):
    size = 2 * n if group == "Sp" else n + 1 if group == "SE" else n
    units = np.eye(size * size).reshape(-1, size, size)
    if group in ("GL", "GL+"):
        return units
    if group == "SL":
        result = [units[i * size + j] for i in range(n) for j in range(n) if i != j]
        # Orthonormal, traceless diagonal Helmert contrasts.
        for k in range(1, n):
            diagonal = np.zeros(n)
            diagonal[:k] = 1
            diagonal[k] = -k
            result.append(np.diag(diagonal / np.sqrt(k * (k + 1))))
        return np.asarray(result)
    if group in ("O", "SO", "SE"):
        result = [(units[i * size + j] - units[j * size + i]) / np.sqrt(2)
                  for i in range(n) for j in range(i + 1, n)]
        if group == "SE":
            result.extend(units[i * size + n] for i in range(n))
        return np.asarray(result)
    # Hamiltonian matrices [[A,B],[C,-A.T]], B and C symmetric.
    result = []
    for i in range(n):
        for j in range(n):
            result.append((units[i * size + j] - units[(n + j) * size + n + i]) / np.sqrt(2))
    for offset in (0, 1):
        for i in range(n):
            for j in range(i, n):
                row, col = (i, n + j) if offset == 0 else (n + i, j)
                other_row, other_col = (j, n + i) if offset == 0 else (n + j, i)
                result.append(units[row * size + col] if i == j else
                              (units[row * size + col] + units[other_row * size + other_col]) / np.sqrt(2))
    return np.asarray(result)


class MatrixChart:
    """Local GL(n), GL+(n), SL(n), O(n), SO(n), Sp(n), or SE(n) chart.

    Coordinates use a Frobenius-orthonormal algebra basis. SE translation
    coordinates carry length units: the caller must choose a consistent scale
    before interpreting a rotation/translation least-squares metric.

    GL and O can represent either determinant component with an explicit
    reference in that component. Values cannot cross components. Encoding and
    decoding reject the negative-real-axis logarithm cut and generators outside
    the principal branch; callers must partition fields into suitable patches.
    """

    def __init__(self, group, n, reference=None):
        if group not in ("GL", "GL+", "SL", "O", "SO", "Sp", "SE"):
            raise ValueError("unknown group family")
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        if group in ("SL", "O", "SO") and n < 2:
            raise ValueError("use n >= 2 for SL, O and SO")
        self.group, self.n = group, int(n)
        self.size = 2 * n if group == "Sp" else n + 1 if group == "SE" else n
        self.basis = _basis(group, n)
        self.dimension = len(self.basis)
        self.basis.setflags(write=False)
        self.reference = np.array(np.eye(self.size) if reference is None else
                                  _real(reference, "reference"), copy=True)
        self._validate(self.reference[None])
        self.reference.setflags(write=False)

    def _validate(self, values):
        if values.ndim != 3 or values.shape[1:] != (self.size, self.size):
            raise ValueError(f"values must have shape (q,{self.size},{self.size})")
        if not len(values):
            return
        if np.any(np.linalg.cond(values) >= 1e12):
            raise ValueError("matrix is singular or too ill-conditioned")
        sign, logdet = np.linalg.slogdet(values)
        if self.group != "GL" and self.group != "O" and np.any(sign <= 0):
            raise ValueError("positive determinant required")
        if self.group in ("SL", "SO", "SE", "Sp") and np.any(np.abs(logdet) > 1e-10):
            raise ValueError("unit determinant required")
        if self.group in ("O", "SO", "SE"):
            rotations = values[:, :self.n, :self.n]
            if np.any(np.linalg.norm(rotations.swapaxes(1, 2) @ rotations - np.eye(self.n), axis=(1, 2)) > 1e-10):
                raise ValueError("orthogonal rotation block required")
        if self.group == "SE":
            last = np.r_[np.zeros(self.n), 1.]
            if not np.allclose(values[:, -1], last, atol=1e-12, rtol=0):
                raise ValueError("SE requires a homogeneous last row")
        if self.group == "Sp":
            zero, identity = np.zeros((self.n, self.n)), np.eye(self.n)
            j = np.block([[zero, identity], [-identity, zero]])
            residual = values.swapaxes(1, 2) @ j @ values - j
            if np.any(np.linalg.norm(residual, axis=(1, 2)) > 1e-10):
                raise ValueError("symplectic constraint violated")

    def encode(self, values):
        values = _real(values, "values")
        self._validate(values)
        result = []
        for value in values:
            relative = np.linalg.solve(self.reference, value)
            eigenvalues = np.linalg.eigvals(relative)
            if np.any(np.abs(np.angle(eigenvalues)) >= np.pi - 1e-7):
                raise ValueError("relative matrix reaches the principal logarithm branch cut")
            logarithm = logm(relative)
            scale = max(1., np.linalg.norm(logarithm))
            if np.linalg.norm(logarithm.imag) > 1e-10 * scale:
                raise ValueError("principal logarithm is not real")
            logarithm = logarithm.real
            coordinates = np.einsum("kij,ij->k", self.basis, logarithm)
            represented = np.einsum("k,kij->ij", coordinates, self.basis)
            if np.linalg.norm(represented - logarithm) > 1e-10 * scale:
                raise ValueError("logarithm lies outside the declared algebra")
            if np.linalg.norm(expm(represented) - relative) > 1e-10 * max(1, np.linalg.norm(relative)):
                raise ValueError("logarithm reconstruction failed")
            result.append(coordinates)
        return np.asarray(result).reshape(len(values), self.dimension)

    def decode(self, coordinates):
        coordinates = _real(coordinates, "coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != self.dimension:
            raise ValueError(f"coordinates must have shape (q,{self.dimension})")
        generators = np.einsum("qk,kij->qij", coordinates, self.basis)
        if len(generators) and np.any(np.abs(np.linalg.eigvals(generators).imag) >= np.pi - 1e-7):
            raise ValueError("generator leaves the principal logarithm branch")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            values = np.asarray([self.reference @ expm(b) for b in generators]).reshape(-1, self.size, self.size)
        if not np.all(np.isfinite(values)):
            raise ValueError("matrix exponential is unrepresentable")
        self._validate(values)
        return values
