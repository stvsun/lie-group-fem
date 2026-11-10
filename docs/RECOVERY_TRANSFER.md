# Variational recovery and full transfer convergence

These studies extend the [nodal interpolation study](CONVERGENCE.md) to the
first stage of recovery and then to a complete source-to-target FE field
transfer. The original interpolation results and their frozen evidence remain
unchanged. The fields are the same independent SO(3), compressible GL⁺(3), and
isochoric GL⁺(3) manufactured fields defined in that protocol.

## Variational recovery

Start with exact matrix samples **only at source integration points**. For the
Lie method, encode those samples in the identity-referenced local chart and
solve

```text
min_C Σq wq ‖Σa Na(xq) Ca − chart(A(xq))‖².
```

For the componentwise method, use the same points, measures, and FE basis to
solve the corresponding problem in all nine matrix entries:

```text
min_B Σq wq ‖Σa Na(xq) Ba − vec(A(xq))‖².
```

Both methods use the global consistent P1 mass matrix, without mass lumping,
regularization, or projection of the componentwise result onto a group. The
recovered continuous fields are evaluated at independent integration points on
the source mesh and compared to the analytic matrix field. The reported error
is a common extrinsic matrix L² norm; it is different from the chart-space
least-squares objective minimized by the Lie method.

For each positively oriented tetrahedron, the consistent mass is
Mₑ = Vₑ(ones(4,4)+I)/20. This small study assembles the sparse global matrix and
solves it with a sparse LU factorization. Positive element volumes and a
conforming P1 mesh give an SPD mass matrix. Relative normal-equation residuals
are recorded for both methods. The production `lie_fem.recover` remains a
dense weighted SVD solver: independent tests compare the study's sparse result
against that API and a separate weighted least-squares componentwise oracle.
This sparse path is specific to the generated study meshes.

## Full mesh-to-mesh transfer

The full study executes all of the following operations using the recovered
source data:

1. Sample the analytic field at source integration points and recover the
   source nodal coordinates/matrix entries as above.
2. Locate each target vertex in the source mesh. Evaluate the recovered source
   P1 field there to obtain the target nodal coordinates/matrix entries.
3. Interpolate the resulting **target FE field** at target integration points,
   decode Lie coordinates, and compare the matrices with analytic truth.

No analytic target nodal values enter the transfer. The target interpolation
uses the same chart reference as recovery; chart coordinates are retained
throughout. Componentwise values remain uncorrected at every step.

The source is the uniform Freudenthal mesh of the unit cube at n = 1, 2, 4, 8.
The target has the same number of subdivisions, reflected x connectivity,
and interior vertex displacement

```text
δX = (0.15/n) sin(πx) sin(πy) sin(πz) (1, −0.6, 0.4).
```

The cube boundary is fixed. Reflection is followed by an orientation correction;
all target element volumes are checked. These meshes have different physical
tetrahedral connectivity; for n ≥ 2 their interior vertices also differ. Source
and target are refined together at comparable resolution. The convergence
abscissa is the source maximum diameter h = √3/n; the actual target maximum
diameter is also recorded. This study is transfer between nonmatching meshes,
not a test of coarsening, a separate target-refinement limit, or arbitrary mesh
search. The structured locator is tested against independent barycentric solves
and exact reproduction of the physical query coordinates.

The report also retains the recovered **source** field evaluated directly at
the target integration points. This intermediate field measures direct
integration-point transfer without imposing a target nodal representation.
Its difference from the final target FE field is reported as a sampled,
normalized transfer increment. That increment can expose extra target
representation error. It is a diagnostic at target sample locations; the total
error is not the sum of recovery and transfer-increment norms.

## Metrics, quadrature, and checks

Both stages use the same relative matrix L² metric as the interpolation study,
along with sampled maximum error, determinant error, minimum determinant,
nonpositive determinant count, and SO(3) orthogonality defect. Isochoric
determinant preservation is checked separately from generic GL⁺(3) positivity.
All tensors are float64 and dimensionless, angles are radians, and coordinates
use normalized length. All reported extrema are sampled, not global bounds.

The protocol is fixed in
[`studies/recovery_protocol.json`](../studies/recovery_protocol.json). Source
loads use a positive 4³-point Duffy–Gauss rule. Primary errors use a different
5³-point rule. On the finest source mesh, two independent checks are performed:

- Recompute source loads with 5³ points, keeping the error-evaluation rule fixed.
- Keep the recovered field fixed and evaluate errors with 6³ points.

For matrix and determinant errors above 10⁻¹², each relative change must be
≤ 10⁻³. Roundoff-scale defects are identified separately. Final-pair matrix
orders must be in the preselected interval [1.8, 2.2]. The relative normal
residual must be below 10⁻¹¹. Lie SO(3) and isochoric sampled group defects must
be below 10⁻¹¹ and all sampled Lie determinants positive. Invalid chart nodes,
inverted mesh elements, and outside-domain queries raise errors. Failed checks
are saved and reported; they never become successful checks through relabeling.

Constant matrix fields must survive both complete transfers. Affine chart fields
must survive recovery and transfer exactly to floating-point accuracy; these
tests use analytic rotations and diagonal exponential stretches. Mesh
geometry, consistent mass, sparse solves, and locator correctness are checked
independently before interpreting nonlinear convergence results.

## Reproduction and limits

```sh
python -m pip install -e '.[test,study]'
python -m pytest
python -m studies.recovery_transfer --output outputs/recovery-transfer-new-run
```

Each run requires a new directory. It retains source quadrature samples, exact
and encoded tensors, the sparse mass system and right-hand sides, recovered
coefficients, target node maps and barycentric weights, final and intermediate
evaluation arrays, metrics, verification runs, exact protocol bytes, source
hashes, and evidence checksums. All raw arrays remain outside Git. PNG and SVG
plots are generated from the verified report. Existing reports can be replotted
with `python -m studies.plot_recovery RUN/report.json --output NEW_PLOT_DIR`.

These are variational **field-recovery and field-transfer** studies. They do not
solve equilibrium, compatibility, a constitutive update, or the papers' coupled
three-field/poromechanics formulations. Group preservation is not a mechanical
conservation law. Accuracy ordering between methods is an observed result for
these fields and references, not a general optimality claim. The full transfer
studied here includes a target nodal FE representation; a solver that stores
only directly transferred integration-point data has a different final stage.
