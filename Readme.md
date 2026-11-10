# Lie-group FEM

A small Python library for **Lie-group interpolation and variational reconstruction
of tensor fields in finite elements**. It provides local SO(3) rotation mapping,
polar GL⁺(3) mapping, and quadrature-to-node recovery in a declared common chart.

This is an independent implementation inspired by the references below. It has
no dependency on PhysicsWorlds, PBX_PM, a constitutive model, or a particular FEM
solver. The runtime dependencies are NumPy and SciPy; calculations use float64.

## Install and run

Python 3.11 or later:

```sh
git clone https://github.com/stvsun/lie-group-fem.git
cd lie-group-fem
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest

python examples/so3_tetrahedron.py
python examples/gl3_remap.py
python examples/variational_recovery.py
```

The repository is private; cloning requires access. Examples create a new folder
under `outputs/`, with raw source/target fields in `fields.npz` and diagnostics
and dependency versions in `report.json`. Use `--output PATH` to select a new
directory. Existing directories are never overwritten. Outputs and environments
are excluded from Git.

## Minimal API

```python
import numpy as np
from lie_fem import SO3Chart, interpolate, recover

chart = SO3Chart(reference=np.eye(3))
rotations = chart.decode([[0, 0, 0], [0, 0, np.pi / 2]])
midpoint = interpolate(rotations, [[0.5, 0.5]], chart)  # 45 degrees

# In a solver: source_values=(nq,3,3), N=(nq,nn), dV=(nq,).
# dV includes quadrature weights and the physical integration Jacobian.
# recovered = recover(source_values, N, dV, chart)
# target_values = recovered.evaluate(N_target)
```

Use `GL3Chart(reference=R0)` for orientation-preserving 3×3 tensor maps. Inputs
always have a leading sample dimension, including a single tensor. Shape-function
rows sum to one. Both charts use the same `interpolate` and `recover` functions.

| Example | FEM operation | Check |
| --- | --- | --- |
| `so3_tetrahedron.py` | Nodal orientations → tetrahedron integration points | Analytic rotation field; orthogonality; componentwise interpolation comparison |
| `gl3_remap.py` | Coarse nodal tensor field → integration points of four refined tetrahedra | Analytic tensor field, positive determinant, isochoric preservation |
| `variational_recovery.py` | Integration points → five shared nodes on two tetrahedra | Exact affine chart field; nonzero residual for unresolved variation; stationarity |

## Mathematical contract

Matrices act on column vectors. Rotation angles are in radians. Deformation
maps and stretches are dimensionless; mesh coordinates use a consistent length
unit and quadrature measures use its volume unit.

For a fixed reference rotation R₀, the SO(3) chart is

```text
r_i = vee(log(R₀ᵀ R_i))
R(x) = R₀ exp(hat(sum_i N_i(x) r_i)).
```

For GL⁺(3), take the right polar decomposition Fᵢ = RᵢUᵢ, with Uᵢ positive
definite, and interpolate the rotation coordinates and the symmetric matrix
Lᵢ = log(Uᵢ). Reconstruct F(x) = R(x) exp(L(x)). Matrix functions reconstruct
the full stretch tensor; individual eigenvectors are never interpolated.

Recovery minimizes the **quadrature-weighted Euclidean error in these chart
coordinates**:

```text
min_C  sum_q dV_q ||sum_a N_qa C_a - c_q||²,
c_q = chart.encode(source_values)[q].
```

The consistent mass equation is NᵀWN C = NᵀWc. The implementation solves the
weighted least-squares design directly using SVD, without squaring its condition
number. It reports the RMS fit error, stationarity residual, and design condition
number. See [docs/METHOD.md](docs/METHOD.md) for coordinate conventions and guards.

## Scope and limits

- `GL3Chart` supports **GL⁺(3), det F > 0**, the component used for admissible
  continuum deformation. Negative-determinant GL(3) maps are rejected.
- The rotation chart is local. Every encoded and reconstructed rotation vector
  must have norm below π − 10⁻⁷. Choose a reference near the data and separate
  patches when necessary. Fields spanning a branch discontinuity can produce
  an inappropriate interpolation even when individual samples pass this guard.
  There is no automatic unwrapping or global intrinsic mean.
- Recovery is variational in the specified chart metric. It does not implement
  the full coupled three-field mechanics formulation of Mota et al., or the
  configurational-force/remeshing solver of Na et al.
- Group membership alone does not enforce compatibility with a displacement
  field, internal-force conservation, equilibrium, or a constitutive yield
  condition. A mechanics solver must enforce its own additional constraints.
- Recover separate material phases or discontinuous fields separately. The
  supplied dense design is intended for small examples and local patches.
  Sparse assembly, mesh search, automatic remeshing, and time integration are
  outside this version.

The tests use analytic rotations, independent matrix exponentials and polar
factors, exact tetrahedron moments, manufactured fields, and a separately
assembled consistent mass equation. They test the declared numerical contract;
they do not establish reproduction of either paper's benchmarks.

## References

1. Mota, A., Sun, W., Ostien, J. T., Foulk III, J. W., & Long, K. N. (2013).
   Lie-group interpolation and variational recovery for internal variables.
   *Computational Mechanics*, *52*(6), 1281–1299.
   [doi:10.1007/s00466-013-0876-1](https://doi.org/10.1007/s00466-013-0876-1).
2. Na, S., Bryant, E. C., & Sun, W. (2019). A configurational force for adaptive
   re-meshing of gradient-enhanced poromechanics problems with history-dependent
   variables. *Computer Methods in Applied Mechanics and Engineering*, *357*,
   112572. [doi:10.1016/j.cma.2019.112572](https://doi.org/10.1016/j.cma.2019.112572).

BibTeX entries are in [references.bib](references.bib). Please cite the relevant
research when building on these ideas. The references provide methodological
context; their text and figures are not distributed with this library.

## License

[MIT](LICENSE), copyright 2026 WaiChing Sun. This license covers this repository's
original code and documentation; dependencies retain their respective licenses.
