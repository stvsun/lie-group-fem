# The eight group families in Table 1

`PositiveRealChart` and `MatrixChart` extend the shared `interpolate` and `recover`
interface. They do not alter `SO3Chart` or the polar `GL3Chart` used by the earlier
convergence studies. Calculations use real float64 values; input tensors always
have a leading sample dimension.

| Paper group | Constructor | Matrix size | Coordinates | Constraint |
| --- | --- | --- | ---: | --- |
| ℝ⁺ | `PositiveRealChart()` | Scalars `(q,1)` | 1 | Positive values |
| GL(n) | `MatrixChart("GL", n, reference=A0)` | n×n | n² | Nonsingular, same determinant component as A0 |
| GL⁺(n) | `MatrixChart("GL+", n)` | n×n | n² | Positive determinant |
| SL(n) | `MatrixChart("SL", n)` | n×n | n²−1 | Unit determinant |
| O(n) | `MatrixChart("O", n, reference=Q0)` | n×n | n(n−1)/2 | Orthogonal, same determinant component as Q0 |
| SO(n) | `MatrixChart("SO", n)` | n×n | n(n−1)/2 | Proper orthogonal |
| Sp(n) | `MatrixChart("Sp", n)` | **2n×2n** | n(2n+1) | AᵀJA=J |
| SE(n) | `MatrixChart("SE", n)` | (n+1)×(n+1) | n(n+1)/2 | Homogeneous proper rigid motion |

The Sp notation follows this paper: J = [[0,I],[-I,0]]. Some other sources call
this real group Sp(2n,ℝ). SL, O and SO require n≥2 in this implementation; the
other matrix constructors accept n≥1. References default to the identity.

## Chart and variational metric

For a fixed reference A₀, encode B = log(A₀⁻¹A) and reconstruct A = A₀ exp(B).
Coordinates are the Frobenius inner products with an orthonormal basis of the
declared algebra. GL uses all matrix units; SL adds orthonormal traceless
diagonal contrasts to the off-diagonal units; O/SO use normalized skew matrices.
The symplectic algebra consists of [[P,Q],[S,−Pᵀ]] with Q and S symmetric.
SE uses a skew rotation block and an independent translation column.

The common recovery routine minimizes the weighted Euclidean error in these
coordinates, equivalent to Frobenius error between the relative logarithms.
For SE, translation carries length units while rotation is dimensionless: choose
and document a length scale before interpreting this combined metric. This is
not a claim of an intrinsic, reference-independent group metric.

The SE exponential includes the rotation/translation coupling; it does not
simply copy the generator's translation column into the final transform.
The positive scalar chart uses log/exp with no reference scaling.

## Locality, components and guards

These are **local charts**, not global logarithms of every group element. A
principal matrix logarithm must be real and lie in the specified algebra.
Encoding rejects relative eigenvalue arguments within 10⁻⁷ of ±π. Decoding
rejects generator eigenvalues whose imaginary parts reach ±(π−10⁻⁷), so it does
not silently wrap rotations. Both directions check constraints, finiteness, and
matrix condition numbers below 10¹². Reconstructed logarithms are checked
against the original relative matrix. Tiny numerical algebra residuals are
tolerated at 10⁻¹⁰; data that fail validation are not repaired.

GL and O have two determinant components. A negative-determinant reference
allows a local chart in the negative component; it does not allow interpolation
between components. Even within GL⁺, a single real principal logarithm is not
available globally. A matrix may have a real nonprincipal logarithm and still
be intentionally rejected by this API. For example, −I₂ has a real skew
logarithm but lies on the principal-log branch cut. The paper's broad wording
about negative eigenvalues is not a general test for existence of a real log.

SPD matrices are a manifold, **not** a vector space and not a group under ordinary
matrix multiplication. The existing polar GL3 chart already maps SPD stretches
through their symmetric matrix logarithms. `MatrixChart("GL+", 3)` instead
uses the direct relative matrix logarithm, so the two constructors define
different interpolation/recovery schemes.

For damage D=1−H, H>0 enforces D<1, but does not by itself enforce D≥0. The
constitutive model must impose the remaining bounds; the same caution applies
to tensor damage, plastic admissibility and mechanical equilibrium.

## Example

```python
import numpy as np
from lie_fem import MatrixChart, PositiveRealChart, interpolate

chart = MatrixChart("SL", 3)
F = np.array([[[1., 2, 0], [0, 1, 0], [0, 0, 1]],
              [[1., 0, 0], [2, 1, 0], [0, 0, 1]]])
mapped = interpolate(F, [[.5, .5], [-.5, 1.5]], chart)
assert np.allclose(np.linalg.det(mapped), 1.)

positive = interpolate([[.9], [.1]], [[.5, .5]], PositiveRealChart())
reflection_patch = MatrixChart("O", 3, reference=np.diag([-1., 1., 1.]))
rigid_motion = MatrixChart("SE", 3)
symplectic = MatrixChart("Sp", 2)  # 4×4 matrices
```

Run `python examples/classical_groups.py --output outputs/groups-new` for all
eight families on a tetrahedron: quadrature samples → recovered nodal
coordinates → new target points. It preserves all source and target arrays and
refuses to overwrite a run. The negative GL/O components are exercised explicitly.
Independent tests cover nilpotent and diagonal exponentials, Rodrigues
rotations, the analytic SE(2) translation integral, symplectic shears, defining
group constraints, affine recovery and invalid inputs.
