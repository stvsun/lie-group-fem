# Implementation and verification contract

## Charts

`SO3Chart` stores a fixed R₀ ∈ SO(3). Its three coordinates are the axial vector
of log(R₀ᵀR), in radians. SciPy's rotation routines evaluate the local exponential
and principal logarithm. Input orthogonality and determinant are checked before
calling them; the library does not silently project arbitrary matrices to SO(3).

`GL3Chart` applies F = RU and stores nine coordinates:

```text
(r_x, r_y, r_z, L_xx, L_yy, L_zz, √2 L_yz, √2 L_xz, √2 L_xy),  L = log U.
```

The six stretch coordinates preserve the Frobenius inner product. Thus the
recovery metric is ||Δr||² + ||ΔL||_F², with equal dimensionless weighting of
rotation and logarithmic stretch. It is a chosen chart metric, not an assertion
of an intrinsic GL(3) distance. SPD stretches form a manifold, not a group under
ordinary matrix multiplication; this is a polar product construction, not a
generic matrix logarithm of F.

SVD gives F = A diag(s) Bᵀ. We reconstruct R = ABᵀ and
L = B diag(log s) Bᵀ. Repeated singular values do not require choosing or
interpolating an eigenframe. Reconstruction uses the symmetric eigendecomposition
of L to evaluate exp L. Tests include isotropic and repeated stretches.

Left multiplication by a common observer rotation Q is equivariant **when the
reference is also changed to QR₀**. Holding R₀ fixed is a different construction.
For a valid chart interpolation, det F(x) = exp(sum_i N_i log det F_i), so nodal
unit determinants are preserved to floating-point error. This does not make an
arbitrary mapped tensor field the gradient of a displacement field.

## Recovery and evaluation

Given source tensors at quadrature locations, the caller supplies target basis
evaluations N at those same locations and positive physical integration measures.
N must have full column rank. Each of the chart coordinates uses the same
weighted least-squares solve. A basis with insufficient support raises an error;
no pseudoinverse fallback or hidden regularization is used for a deficient basis.

For stable unit scaling, weights are normalized by their maximum and sum. The
reported RMS is sqrt(sum_q dV_q ||(NC − c)_q||² / sum_q dV_q). The stationarity
residual is ||Nᵀ diag(dV/sum dV) (NC − c)||_F. The design condition number is
that of diag(sqrt(dV/sum dV)) N, not the squared mass-matrix condition number.

`RecoveredField` retains nodal coordinates and the chart. `evaluate(N_target)`
interpolates those coordinates directly. It checks the reconstructed chart
domain each time, including for signed higher-order weights.

Tetrahedron helpers implement positively oriented P1 elements and the symmetric
four-point degree-two rule. They reject outside evaluation points. They provide
no point locator; the remapping example explicitly maps points within a known
source tetrahedron. Integration measures are reference volumes in the examples.

## Numerical guards

- Real, finite float64 arrays; explicit batch shapes; no in-place source edits.
- Rotation orthogonality error in Frobenius norm and determinant error ≤ 10⁻¹⁰.
- Rotation-vector norm < π − 10⁻⁷ for encoding and decoding. A suitable local
  reference and branch-consistent data remain the caller's responsibility.
- Deformation/stretch condition number < 10¹²; det F strictly positive.
- Recovery-design and tetrahedron-edge condition number < 10¹².
- Shape-function partition-of-unity absolute tolerance 10⁻¹².
- Exponent overflow/underflow and nonpositive reconstructed determinants fail
  explicitly. These numerical guards do not promise accuracy near their limits.

## Source boundaries

The [2013 publisher abstract](https://link.springer.com/article/10.1007/s00466-013-0876-1)
motivates admissible-space interpolation and variational internal-variable
recovery. The [2019 paper](https://doi.org/10.1016/j.cma.2019.112572) motivates
history-variable mapping during poromechanics remeshing. Its
[author manuscript](https://www.osti.gov/servlets/purl/1801238) is also available.
Bibliographic details were checked against publisher and author sources on
2026-11-10. This repository defines and tests its own restricted algorithm above;
it makes no claim of a complete paper reproduction or mechanical equilibrium.
