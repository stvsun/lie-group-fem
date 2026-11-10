# Recovery and transfer results — 2026-11-10

Selected run: `outputs/recovery-transfer-20261110-01`. All 132 declared checks
passed: 48 source/evaluation quadrature comparisons, 48 primary normal-equation
checks, 24 Lie group-constraint checks, and 12 final-pair matrix-order checks.
The [metrics CSV](recovery-transfer-metrics.csv) includes every mesh level and
the [machine-readable receipt](recovery-transfer-receipt.json) records the full
checks, fixed protocol, environment, source hashes, and evidence hashes.

## Finest meshes

Both meshes have 729 nodes and 3072 elements. Source maximum diameter is
0.21650635094610965; target maximum diameter is 0.222313196299801. The target
has different connectivity and displaced interior nodes. Every primary
error evaluation uses 384000 quadrature points.

| Stage | Field | Lie relative L² | Componentwise relative L² | Last order, Lie / componentwise |
| --- | --- | ---: | ---: | ---: |
| Recovery | SO(3) | 2.111174e-03 | 2.340896e-03 | 2.018 / 2.035 |
| Recovery | GL⁺(3), compressible | 2.191283e-03 | 2.419043e-03 | 2.020 / 2.040 |
| Recovery | GL⁺(3), isochoric | 2.191554e-03 | 2.415558e-03 | 2.019 / 2.034 |
| Full transfer | SO(3) | 3.471185e-03 | 3.874878e-03 | 2.001 / 1.994 |
| Full transfer | GL⁺(3), compressible | 3.465038e-03 | 4.029733e-03 | 2.004 / 2.000 |
| Full transfer | GL⁺(3), isochoric | 3.547197e-03 | 4.017151e-03 | 2.004 / 1.996 |

The observed final-pair orders are 2.0182–2.0396 for recovery and
1.9939–2.0044 for full transfer. The higher source quadrature changes the
meaningful matrix/determinant errors by at most 4.25137×10⁻⁸ relative; the
higher evaluation quadrature changes them by at most 2.01841×10⁻⁹. Both are
below the preselected 10⁻³ tolerance. Maximum primary normal-equation relative
residual is 3.20554×10⁻¹⁶.

## Group constraints and accuracy

For SO(3), the finest orthogonality RMS is 3.29586×10⁻¹⁶ (Lie) versus
3.83106×10⁻³ (componentwise) after recovery, and 3.28994×10⁻¹⁶ versus
6.18886×10⁻³ after full transfer. For isochoric GL⁺(3), the determinant relative
L² error is 9.17442×10⁻¹⁶ versus 2.26294×10⁻³ after recovery, and
9.17880×10⁻¹⁶ versus 3.53124×10⁻³ after full transfer.

Every sampled determinant remains positive for both methods at every level;
this is an observation, not a general closure property of componentwise
mapping. Group preservation does not enforce equilibrium or displacement
compatibility. The Lie method has lower integrated matrix error in these runs,
but its compressible recovery sampled maximum error is 0.013583 versus
0.013306 for componentwise recovery. There is no universal pointwise ranking.

## Added target-representation error

The full-transfer stage includes source integration-point recovery, mapping to
target nodes, and interpolation of the target FE field. At the same target
quadrature points, the report additionally saves direct evaluation of the
recovered source field and the normalized difference introduced by target
representation. For SO(3) at n=8, those direct errors are 2.12835×10⁻³ (Lie)
and 2.35977×10⁻³ (componentwise), versus final errors 3.47119×10⁻³ and
3.87488×10⁻³. The transfer-increment norms are 3.87634×10⁻³ and 4.18725×10⁻³.
These norms are not additive; the increment is a sampled diagnostic crossing
source element boundaries, not a separately quadrature-certified continuous
norm. See all analogous GL⁺(3) diagnostics in the CSV.

## Reproduction and preserved evidence

Use the [protocol](RECOVERY_TRANSFER.md) commands with a new output directory.
The selected raw run is retained locally, outside Git, with source quadrature
values, sparse mass system, recovered coefficients, target barycentric maps,
exact and transferred matrices, intermediate direct-transfer values, verification
runs, and checksums. No existing interpolation evidence was rewritten.

Python 3.14.3, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2; macOS arm64.
The run preceded its implementation commit, so its receipt includes the baseline
Git commit and dirty-tree flag alongside exact source hashes. Plots are copied
from that run, not recomputed from rounded table entries.
