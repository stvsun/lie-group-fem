# Interpolation convergence protocol

This study measures **nodal interpolation**, the second stage of the conceptual
transfer diagram. The nodal values are exact samples of manufactured fields.
Integration-point recovery, mechanics re-equilibration, constitutive evolution,
and an end-to-end remeshing error are not measured here.

## Fixed comparison

The unit cube is divided into n³ cubes, each split into six conforming,
positively oriented Freudenthal tetrahedra. The four levels n = 1, 2, 4, 8 have
6, 48, 384, and 3072 P1 elements. The maximum tetrahedron diameter is
h = √3/n. Coordinates use normalized length, rotations radians, and tensor
maps and stretches are dimensionless. All calculations use float64.

Both methods receive exactly the same nodal matrices, barycentric weights,
physical evaluation points, and integration measures. The Lie method uses the
library's common chart with the identity reference at every mesh level; the
componentwise method computes ΣₐNₐAₐ directly, without polar repair or projection.
The study batches node encoding and point decoding; tests compare that batching
against the public `interpolate` function element by element.

## Manufactured fields

For (x,y,z) ∈ [0,1]³, define the standard right-handed axis rotations and

```text
a = 0.2 + 1.1 sin(πx/2) cos(πy/2) + 0.25 z²
b = −0.2 + 0.65 sin(πy/2) + 0.2 xz
c = 0.15 + 0.5 sin(πz/2) + 0.15 xy
R = Rz(a) Ry(b) Rx(c)

Q  = Rz(0.3 + 0.4x + 0.15 sin(πz)) Ry(0.2y)
s1 = 0.25 sin(πx/2) + 0.12 yz
s2 = −0.18 cos(πy/2) + 0.10 xz
U  = Q diag(exp(s1), exp(s2), exp(s3)) Qᵀ
F  = RU.
```

The SO(3) case uses R. The compressible GL⁺(3) case uses
s3 = 0.10x − 0.08y + 0.12 sin(πz); the isochoric case uses s3 = −s1 − s2.
Thus the GL⁺(3) ground-truth determinant is exp(s1+s2+s3), and is exactly one
in the isochoric case. The reference matrices are built from closed-form
trigonometric rotations and scalar exponentials, independently of the production
SO(3) logarithm/exponential and GL⁺(3) polar routines. These fields are nonlinear
in space with noncommuting rotations and spatially varying principal directions;
they are not chosen to be exactly affine in the interpolation chart.

## Metrics and integration

The primary common accuracy metric is

```text
E = sqrt(Σq wq ‖Ah(xq) − A(xq)‖F² / Σq wq ‖A(xq)‖F²).
p = log(Ecoarse / Efine) / log(hcoarse / hfine).
```

It is an extrinsic matrix norm that applies equally to an off-manifold
componentwise result. No invalid matrix is converted to an SO(3) logarithm.
We also record sampled maximum matrix error, relative L² determinant error,
sampled minimum determinant and count of nonpositive determinants. For SO(3),
we record RMS and sampled maximum ‖RᵀR − I‖F. Sampled extrema are not certified
global bounds. Orthogonality is not a GL⁺(3) requirement, so it is not reported
as an admissibility metric for F.

Each tetrahedron uses a positive 4×4×4 Gauss–Legendre rule through the Duffy map
x=u, y=(1−u)v, z=(1−u)(1−v)w, with Jacobian (1−u)²(1−v). The 64-point rule
integrates all degree-five tetrahedron polynomials exactly. Tests check every
monomial through that degree against its analytic simplex integral.

The finest mesh is evaluated again with a 5×5×5 rule, checked independently
through degree seven. Before running, the acceptable relative quadrature change
was set to 10⁻³ for matrix and determinant L² errors above the 10⁻¹² roundoff
threshold. The preselected last-pair matrix convergence-order interval is
[1.8, 2.2]. Checks are reported explicitly; a failed study retains all raw data
and exits with an error. The frozen settings are in
[`studies/protocol.json`](../studies/protocol.json).

## Interpretation and separate admissibility example

Second-order convergence is consistent with smooth-field P1 interpolation.
The chart construction does not imply a universally smaller error constant,
better pointwise accuracy, or higher convergence order. Reference choice and
branch consistency matter. The present fields remain in the chosen chart.

Positive determinants at sampled points do not prove closure of GL⁺(3) under
componentwise interpolation. Separately from the refinement study, the report
includes A=diag(3,0.1,1) and B=diag(−0.1,−3,1). Both determinants are 0.3, but
their equal componentwise average has determinant −2.1025. With reference
Rz(π/2), the Lie interpolation stays inside its local chart and has determinant
0.3. This is an algebraic counterexample, not a failed mesh-study sample.

## Reproduction and evidence

```sh
python -m pip install -e '.[test,study]'
python -m pytest
python -m studies.interpolation --output outputs/convergence-new-run
```

An existing output directory is rejected. Each run preserves the exact protocol
bytes; meshes; nodal source matrices and coordinates; all quadrature coordinates,
weights, exact matrices, and both interpolants; a CSV table; a full JSON report;
and SHA-256 hashes of the evidence. Source-file hashes, the baseline Git commit,
working-tree status, and dependency versions identify the implementation even
when run before committing changes. Numerical records are never smoothed or
replaced by trend lines. Plot reference slopes are labeled explicitly.

`python -m studies.plot_convergence RUN/report.json --output NEW_PLOT_DIR`
replots an existing successful report. PNG and SVG versions are produced.
Raw runs stay outside Git. The selected figures, compact metrics table, and
[published result receipt](CONVERGENCE_RESULTS.md) are documentation artifacts.
