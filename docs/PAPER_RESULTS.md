# Paper-example result receipt

Run: `outputs/paper-2013-20261116-07`. All 14 analytic verification checks pass.
**Exact agreement with every printed result is not established.** The billet
problem is excluded at the user’s request; no Albany is used.

The [protocol and assumptions](PAPER_REPLICATION.md), [full report](paper-results.json),
[convergence CSV](paper-ring-convergence.csv), and [hash receipt](paper-receipt.json)
are retained. Raw arrays remain in the run directory outside Git.

## Table comparisons

All four Table 3 group cases agree to the printed precision. Table 2 agrees
except for the SO(3) extrapolation entry (row 3, column 2): the printed
−1.50 differs from the arithmetic result −0.50. This mismatch is retained.

## Scalar projection

Global nodal error against X/4+1/2: 1.332268e-15.
Local sample error: 0.000000e+00.
Global L² error: 0.707106781186547, agreeing with √(1/2).

## Ring interpolation

L=1, centered beam, 8×2×1 coarse cells and 100×25×1 fine cells.
Roundoff-level results are compared with a 10⁻¹¹ tolerance, not bitwise equality.

| Method | Computed maximum | Printed maximum | Agreement |
| --- | ---: | ---: | --- |
| direct | 1.18499916e-01 | 1.18500000e-01 | Yes (rounding/roundoff) |
| mixed | 1.12127029e-15 | 1.21000000e-15 | Yes (rounding/roundoff) |
| lie | 5.34333058e-03 | 1.77000000e-02 | No |
| matrix_log | 2.33502542e+00 | 2.20000000e+00 | No |

The complex no-polar diagnostic has maximum absolute imaginary entry
1.546349e+00; imaginary data are not discarded.

## Ring recovery

Each computed value is followed by the printed Table 7 value in parentheses.
The independent quadrature column is an additional audit, absent from Table 7.

| Method | Nodal minimum (paper) | Nodal maximum (paper) | Source-rule L² (paper) | Independent L² |
| --- | ---: | ---: | ---: | ---: |
| nearest | 1.94343e-02 (6.340e-02) | 2.59188e-01 (1.410e-01) | 6.79180e-03 (8.700e-03) | 7.06171e-03 |
| extrapolation | 6.31585e-02 (6.620e-02) | 7.94675e-02 (6.960e-01) | 7.45373e-04 (5.080e-02) | 2.19363e-03 |
| direct | 6.55588e-02 (1.680e-02) | 8.31784e-02 (1.990e-02) | 6.81492e-04 (2.010e-04) | 2.17756e-03 |
| direct_F | 6.55588e-02 (1.680e-02) | 8.31784e-02 (1.990e-02) | 6.81492e-04 (2.010e-04) | 2.17756e-03 |
| mixed | 2.42539e-16 (5.380e-16) | 8.40916e-15 (2.450e-13) | 6.68236e-17 (3.300e-16) | 7.20336e-17 |
| lie | 2.76067e-03 (7.800e-03) | 3.61233e-03 (8.700e-03) | 1.33700e-05 (3.350e-16) | 9.14577e-05 |

Only the mixed-method row agrees at the declared roundoff tolerance. The
remaining entries are discrepancies, not successful reproduction. Unspecified
mesh layout and length scale are documented; no parameters were fitted to these numbers.

## Convergence and verification

Independent 5³ quadrature gives the following final hₓ orders:

- direct: 1.999089.
- lie: 1.999836.

The mixed method remains at roundoff at every refinement level. Table 3
rounding, scalar exactness/integral, mixed ring exactness, stationarity, exact
factors and convergence rates account for the 14 analytic checks. Library
tests additionally use independent exponential, rotation, SE and symplectic oracles.

## Run history

Runs 01 and 02 stopped while serializing NumPy boolean flags; their raw arrays
were kept. Runs 03–05 completed and were superseded by branch-guard consistency,
provenance and plot-label improvements. Run 06 was superseded by compact SVG heatmap export.
No previous run or frozen study was overwritten.

Versions: python 3.14.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2.
