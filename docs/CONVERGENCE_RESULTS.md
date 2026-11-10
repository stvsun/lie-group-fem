# Convergence results — 2026-11-10

Selected run: `outputs/convergence-20261110-01`. All 18 declared checks passed:
12 quadrature comparisons and six last-pair matrix-order checks. The raw run
is retained locally and excluded from Git. The compact numerical record is
[convergence-metrics.csv](convergence-metrics.csv); the machine-readable source,
environment, check and figure receipt is
[convergence-receipt.json](convergence-receipt.json).

## Finest mesh

| Field | Lie-group relative L² | Componentwise relative L² | Last order, Lie / componentwise |
| --- | ---: | ---: | ---: |
| SO(3) | 4.501856e-03 | 5.077178e-03 | 1.988 / 1.977 |
| GL⁺(3), compressible | 4.673236e-03 | 5.299977e-03 | 1.987 / 1.976 |
| GL⁺(3), isochoric | 4.682337e-03 | 5.245437e-03 | 1.988 / 1.977 |

The maximum relative change in matrix L² error under 64 → 125 quadrature points
per tetrahedron is 1.06461×10⁻⁶. This is 0.000106461%, below the specified 0.1%
threshold. Roundoff-scale determinant defects are excluded from meaningful
relative quadrature comparisons. The remaining determinant-error checks pass.

## Interpretation

The final matrix orders are 1.9756–1.9880, consistent with second-order P1
convergence. Lie-group interpolation has a smaller integrated matrix error for
these manufactured fields, but is not uniformly more accurate pointwise: in the
compressible case its sampled maximum error is 0.018619 versus 0.018397 for
componentwise interpolation. The study makes no universal error-ranking claim.

At n=8, the SO(3) orthogonality RMS is 3.28434×10⁻¹⁶ versus 9.61096×10⁻³;
the isochoric determinant relative L² error is 9.92388×10⁻¹⁶ versus 6.59366×10⁻³.
All sampled determinants are positive for both methods at every mesh level.
The separate two-matrix counterexample has positive input determinants 0.3,
componentwise determinant −2.1025, and Lie-interpolated determinant 0.3.

These are local-chart interpolation results. They do not test a global intrinsic
mean, recovery convergence, mechanical equilibrium, or the papers' benchmarks.

## Evidence and environment

Python 3.14.3, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2; macOS arm64.
The receipt records the baseline commit and dirty working-tree flag because the
study ran before its implementation was committed. Exact source SHA-256 hashes
identify the executed numerical implementation. The original protocol and raw
arrays remain unchanged in the run directory; published plots were copied from
that run. Re-running uses a new directory, never a replacement of those bytes.
