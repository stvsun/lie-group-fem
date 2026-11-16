# Mota et al. (2013): analytic examples and reproduction limits

Source: the user-supplied `LieGroupLieAlg.pdf`, DOI
[10.1007/s00466-013-0876-1](https://doi.org/10.1007/s00466-013-0876-1).
The source PDF's SHA-256 is
`a00ff4986cd533f96cc7932628aaf773d815bfd90fa6041fdcc9d61ae5cb41ad`.
The PDF is not redistributed. Table 1's eight groups are implemented in
[the classical-group API](CLASSICAL_GROUPS.md).

The implementation covers Tables 2–3 and Sections 6.1, 6.2, 6.2.1 and 6.2.2.
**Section 6.3's billet problem is excluded at the user's request. No Albany
dependency, mesh generation or billet simulation is included.**

This is a replay of the specified equations, not a claim that every published
number has been reproduced. Numerical verification and comparison with the
paper are separate report fields. Published discrepancies remain visible in
[the result receipt](PAPER_RESULTS.md) and its JSON data.

## Reproduction command and evidence

```sh
python -m pip install -e '.[test,study]'
python -m pytest
python -m studies.paper_examples --output outputs/paper-new-run
python examples/classical_groups.py --output outputs/groups-new-run
```

Every destination must be new. The paper runner saves its frozen protocol,
source hashes, software versions, raw NPZ arrays, metrics, printed comparisons,
PNG/SVG plots and a SHA-256 manifest. Matrix-log failures of real-valuedness
are retained as complex data in the deliberately unrestricted no-polar
diagnostic; the library's real chart would reject those samples. Prior raw
studies and their frozen source bytes are unchanged. Raw runs remain outside
Git; selected reports and figures are published for review.

## Tables 2 and 3

Use the paper's four source pairs (ℝ⁺, GL⁺(3), SL(3), SO(3)), linear weights
N₁=(1−ξ)/2, N₂=(1+ξ)/2, and evaluation points ξ=0 and ξ=2. Table 2 uses matrix
components; Table 3 uses exp(ΣNᵢlog Zᵢ), without polar decomposition. All Table 3
entries agree to their printed two-decimal precision (absolute tolerance 0.00501).

Table 2 contains an apparent arithmetic typo. Its SO(3) extrapolation prints
−1.50 in row 3, column 2. The supplied matrices and ξ=2 weights give
−0.5×1+1.5×0=−0.50. The raw report retains the printed −1.50 and flags this
comparison as a mismatch. It does not modify the source inputs to reproduce it.

## Section 6.1: discontinuous scalar projection

Use [-1,1]³ with 2×2×2 affine Q1 hexahedra and 2³ Gauss points in each cell.
The source is z=X+1 for X<0 and z=X for X≥0. Solve the consistent global mass
equation and, separately, eight independent element projections. The global
field is X/4+1/2; the element fields reproduce each branch exactly, including
the discontinuity. The independently integrated global squared L² error is 1/2.
The plot shows the field along X; it is constant in Y and Z.

## Section 6.2: ring interpolation

The exact fields are Eqs. 6.2–6.4 with R=L/(2π):

```text
theta = X/R
rotation = rotation_about_z(theta)
stretch  = diag((R−Y)/R, 1, 1)
F        = rotation @ stretch
```

The beam has dimensions L×L/16×L/16. We explicitly set L=1 and center its
reference box at the origin. L is not numerically fixed in the paper. Matrix
errors are dimensionless; absolute volume L² errors scale with L^(3/2).

The paper states 16 coarse and 2500 fine elements but does not give directional
counts or original mesh files. We choose **8×2×1 and 100×25×1**, extending the
aspect ratio 4 and single Z layer specified in Section 6.2.1 to these meshes.
This is a declared assumption, not a recovered original mesh. Fields from the
coarse nodes are evaluated at the fine mesh nodes using the coarse Q1 basis.

The four schemes are direct R/direct S; Lie R/direct S; Lie R/Lie S; and the
direct matrix logarithm of F without polar decomposition. In the polar schemes
we use the exact factors above. Rotation coordinates use the **continuous
analytic angle X/R from Eq. 6.7** rather than re-encoding ±π at the beam ends
with a principal logarithm. This supplied lift belongs to this analytic
benchmark; it is not a general automatic unwrapping algorithm or a relaxation
of the library's branch guard.

The no-polar diagnostic uses SciPy's complex principal logarithm and exponential,
retaining imaginary parts in its outputs and errors. The paper gives neither
the original matrix-log implementation nor a reproducible branch-selection
rule. Its printed maximum is therefore compared, not asserted as an oracle.

## Section 6.2.1: convergence

Refine meshes (4k,k,1), k=1,2,4,8,16: 4 through 1024 elements. These satisfy the
paper's starting mesh and aspect ratio. The paper does not list every refinement
level, so this list is fixed explicitly in the protocol. Compute absolute volume
L² errors for F, R and S using both the source-style 2³ rule and an independent
5³ Gauss rule. Reports retain both, and plots use the independent rule. They are
field interpolation errors, not errors from solving a bending equilibrium problem.

Direct interpolation is exact for the affine stretch; the mixed method is exact
for the affine rotation angle and stretch; both-Lie interpolation introduces
stretch error through log(1−2πY). The nonzero errors approach second order in
axial element size. Since the number of elements grows as hₓ⁻², this is also
first order in inverse element count. The two rate descriptions are consistent.

## Section 6.2.2: recovery

Sample exact F at the coarse grid's 2³ Gauss points. Implement all six described
methods: nearest-point assignment and incident-element averaging; Q1
extrapolation from Gauss points and the same averaging; consistent variational
recovery with polar direct, no-polar direct F, polar mixed, and polar both-Lie
representations. Consistent sparse mass solves check stationarity. No mass
lumping, implicit regularization or componentwise projection back to a group
is applied. The extrapolation rule is obtained by inverting the Q1 basis at the
eight Gauss points.

Retain minimum/maximum nodal Frobenius errors and E_F evaluated back at the
source Gauss points, as in Table 7. **Also evaluate a separate 5³ quadrature
rule.** A near-zero error at the fitting points can coexist with nonzero errors
elsewhere. A dedicated one-Y-cell test demonstrates this for the both-Lie
scheme. The chosen two-Y-cell mesh does not give the same collocation
exactness, and we report the difference from Table 7 instead of hiding it.

The mixed method agrees with the exact field to roundoff. Most Table 7 values
do not agree under the declared mesh and length scale. Original mesh
coordinates, scale, quadrature and recovery scripts would be needed to resolve
those differences. We do not claim that mesh uncertainty alone explains every
discrepancy, or tune those inputs to the published errors.

## Verification boundary

Fourteen runner checks cover Table 3 rounding, exact scalar projections and
error integral, mixed ring exactness at nodes and independent evaluation points,
projection stationarity, exact factors and asymptotic rates. Unit tests also
check Q1 partition of unity, affine coordinates, exact box moments, the scalar
global projection, the known table discrepancy and the collocation caveat.
Passing these checks verifies the implemented algorithms. It does not convert
the explicitly failed printed comparisons into successful paper reproduction.
