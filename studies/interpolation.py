"""P1 interpolation convergence with independent manufactured matrix fields.

Run from the repository root: python -m studies.interpolation --output NEW_DIR
The comparison never projects or repairs the componentwise interpolant.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy

from lie_fem import GL3Chart, SO3Chart

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lie", "componentwise")


def cube_mesh(n):
    """Conforming Freudenthal tetrahedra; n cubes along each axis."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    vertices = np.array(list(itertools.product(range(n + 1), repeat=3)), dtype=np.float64) / n
    cells = []
    strides = np.array([(n + 1)**2, n + 1, 1])
    for origin in itertools.product(range(n), repeat=3):
        for permutation in itertools.permutations(range(3)):
            corners = [np.array(origin)]
            for axis in permutation:
                step = np.eye(3, dtype=int)[axis]
                corners.append(corners[-1] + step)
            nodes = np.array(corners) @ strides
            v = vertices[nodes]
            if np.linalg.det((v[1:] - v[0]).T) < 0:
                nodes[[1, 2]] = nodes[[2, 1]]
            cells.append(nodes)
    return vertices, np.asarray(cells, dtype=np.int64)


def tetrahedron_rule(order):
    """Positive tensor Gauss rule through a Duffy map of the reference tet.

    x=u, y=(1-u)v, z=(1-u)(1-v)w, Jacobian=(1-u)^2(1-v).
    The 4^3 rule integrates all tetrahedron polynomials of total degree <= 5.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 2:
        raise ValueError("order must be an integer >= 2")
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    index = np.array(list(itertools.product(range(order), repeat=3)))
    u, v, w = nodes[index].T
    x, y, z = u, (1-u)*v, (1-u)*(1-v)*w
    shape = np.column_stack([1-x-y-z, x, y, z])
    measures = np.prod(weights[index], axis=1) * (1-u)**2 * (1-v)
    return shape, measures


def axis_rotation(angle, axis):
    """Closed-form rotations; does not call the production chart or SciPy."""
    angle = np.asarray(angle, dtype=np.float64)
    result = np.zeros(angle.shape + (3, 3))
    i, j = (axis + 1) % 3, (axis + 2) % 3
    result[..., axis, axis] = 1
    result[..., i, i] = result[..., j, j] = np.cos(angle)
    result[..., j, i], result[..., i, j] = np.sin(angle), -np.sin(angle)
    return result


def manufactured(points, case):
    """Nonlinear 3D fields; independent trigonometric/spectral ground truth."""
    x, y, z = np.asarray(points, dtype=np.float64).T
    a = .2 + 1.1*np.sin(np.pi*x/2)*np.cos(np.pi*y/2) + .25*z*z
    b = -.2 + .65*np.sin(np.pi*y/2) + .2*x*z
    c = .15 + .5*np.sin(np.pi*z/2) + .15*x*y
    rotation = axis_rotation(a, 2) @ axis_rotation(b, 1) @ axis_rotation(c, 0)
    if case == "so3":
        return rotation
    if case not in ("gl3_compressible", "gl3_isochoric"):
        raise ValueError("unknown manufactured field")
    s1 = .25*np.sin(np.pi*x/2) + .12*y*z
    s2 = -.18*np.cos(np.pi*y/2) + .10*x*z
    s3 = .10*x - .08*y + .12*np.sin(np.pi*z) if case == "gl3_compressible" else -s1-s2
    basis = axis_rotation(.3 + .4*x + .15*np.sin(np.pi*z), 2) @ axis_rotation(.2*y, 1)
    stretch = (basis * np.exp(np.column_stack([s1, s2, s3]))[:, None, :]) @ basis.transpose(0, 2, 1)
    return rotation @ stretch


def chart_for(case):
    return SO3Chart() if case == "so3" else GL3Chart()


def sample_interpolants(nodal_values, nodal_coordinates, cells, vertices, shape, chart):
    """Same FE samples for both methods; cache node encoding, batch decoding."""
    points = np.einsum("qa,eai->eqi", shape, vertices[cells])
    coordinates = np.einsum("qa,eak->eqk", shape, nodal_coordinates[cells])
    lie = chart.decode(coordinates.reshape(-1, coordinates.shape[-1]))
    componentwise = np.einsum("qa,eaij->eqij", shape, nodal_values[cells]).reshape(-1, 3, 3)
    return points.reshape(-1, 3), lie, componentwise


def level(case, n, order, raw_directory=None, batch_elements=256):
    """Return metrics, retaining every sampled tensor if raw_directory is given."""
    vertices, cells = cube_mesh(n)
    shape, reference_weights = tetrahedron_rule(order)
    chart = chart_for(case)
    nodal = manufactured(vertices, case)
    coordinates = chart.encode(nodal)
    if raw_directory is not None:
        raw_directory = Path(raw_directory)
        raw_directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(raw_directory / "mesh-and-source.npz", vertices=vertices, cells=cells,
                            nodal_values=nodal, nodal_coordinates=coordinates, reference=chart.reference,
                            quadrature_shape=shape, reference_weights=reference_weights)
    stats = {method: dict(error2=0., norm2=0., determinant_error2=0., determinant_norm2=0.,
                         orthogonality2=0., max_error=0., max_orthogonality=0.,
                         max_determinant_error=0., min_det=float("inf"), nonpositive=0) for method in METHODS}
    total_volume = 0.
    for start in range(0, len(cells), batch_elements):
        batch = cells[start:start + batch_elements]
        points, lie, componentwise = sample_interpolants(nodal, coordinates, batch, vertices, shape, chart)
        truth = manufactured(points, case)
        tetrahedra = vertices[batch]
        jacobians = np.linalg.det((tetrahedra[:, 1:] - tetrahedra[:, :1]).transpose(0, 2, 1))
        weights = (jacobians[:, None] * reference_weights).ravel()
        truth_det = np.linalg.det(truth)
        total_volume += float(weights.sum())
        for method, approximate in zip(METHODS, (lie, componentwise)):
            error2 = np.sum((approximate-truth)**2, axis=(1, 2))
            determinant = np.linalg.det(approximate)
            orthogonality2 = np.sum((approximate.transpose(0, 2, 1) @ approximate - np.eye(3))**2, axis=(1, 2))
            values = stats[method]
            values["error2"] += float(weights @ error2)
            values["norm2"] += float(weights @ np.sum(truth**2, axis=(1, 2)))
            values["determinant_error2"] += float(weights @ (determinant-truth_det)**2)
            values["determinant_norm2"] += float(weights @ truth_det**2)
            values["orthogonality2"] += float(weights @ orthogonality2)
            values["max_error"] = max(values["max_error"], float(np.sqrt(error2.max())))
            values["max_orthogonality"] = max(values["max_orthogonality"], float(np.sqrt(orthogonality2.max())))
            values["max_determinant_error"] = max(values["max_determinant_error"], float(np.max(np.abs(determinant-truth_det))))
            values["min_det"] = min(values["min_det"], float(determinant.min()))
            values["nonpositive"] += int(np.count_nonzero(determinant <= 0))
        if raw_directory is not None:
            np.savez_compressed(raw_directory / f"samples-{start:06d}.npz", points=points, measures=weights,
                                exact=truth, lie=lie, componentwise=componentwise,
                                element_indices=np.arange(start, start+len(batch)))
    rows = []
    for method, values in stats.items():
        rows.append(dict(
            case=case, method=method, n=n, h=float(np.sqrt(3)/n), nodes=len(vertices),
            elements=len(cells), samples=len(cells)*len(shape), quadrature_order=order, volume=total_volume,
            relative_l2=float(np.sqrt(values["error2"]/values["norm2"])),
            sampled_max_error=values["max_error"],
            relative_determinant_l2=float(np.sqrt(values["determinant_error2"]/values["determinant_norm2"])),
            orthogonality_rms=float(np.sqrt(values["orthogonality2"]/total_volume)) if case == "so3" else None,
            sampled_max_orthogonality=values["max_orthogonality"] if case == "so3" else None,
            sampled_max_determinant_error=values["max_determinant_error"],
            sampled_min_determinant=values["min_det"], nonpositive_samples=values["nonpositive"],
        ))
    return rows


def add_orders(rows):
    previous = {}
    for row in rows:
        key = row["case"], row["method"]
        old = previous.get(key)
        row["observed_order"] = None if old is None else float(np.log(old["relative_l2"]/row["relative_l2"])/np.log(old["h"]/row["h"]))
        previous[key] = row


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def admissibility_counterexample():
    """A separate algebraic diagnostic, not a mesh convergence data point."""
    reference = axis_rotation(np.array(np.pi/2), 2)
    values = np.array([np.diag([3., .1, 1.]), np.diag([-.1, -3., 1.])])
    chart = GL3Chart(reference)
    coordinates = chart.encode(values)
    lie = chart.decode(coordinates.mean(axis=0, keepdims=True))[0]
    componentwise = values.mean(axis=0)
    return dict(source=values.tolist(), reference=reference.tolist(), weights=[.5, .5],
                lie=lie.tolist(), componentwise=componentwise.tolist(),
                source_determinants=np.linalg.det(values).tolist(),
                lie_determinant=float(np.linalg.det(lie)), componentwise_determinant=float(np.linalg.det(componentwise)))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_bytes = (ROOT / "studies/protocol.json").read_bytes()
    (output / "protocol.json").write_bytes(protocol_bytes)
    protocol = json.loads(protocol_bytes)
    rows, verification, checks = [], [], []
    for case in protocol["cases"]:
        for n in protocol["levels"]:
            current = level(case, n, protocol["quadrature_order"], output / "raw" / case / f"n{n:03d}", protocol["raw_batch_elements"])
            rows.extend(current)
            print(f"{case}, n={n}: " + ", ".join(f"{r['method']} L2={r['relative_l2']:.6e}" for r in current), flush=True)
        fine = level(case, protocol["levels"][-1], protocol["verification_quadrature_order"], output / "raw" / case / "quadrature-check", protocol["raw_batch_elements"])
        verification.extend(fine)
        for base, checked in zip(rows[-2:], fine):
            for metric in ("relative_l2", "relative_determinant_l2"):
                # Roundoff-scale invariant defects have no meaningful relative difference.
                reference = checked[metric]
                relative = abs(base[metric]-reference)/max(reference, 1e-12)
                passed = relative <= protocol["quadrature_relative_tolerance"] or max(base[metric], reference) < 1e-12
                checks.append(dict(check="quadrature", case=case, method=base["method"], metric=metric,
                                   relative_change=relative, passed=passed))
    add_orders(rows)
    lower, upper = protocol["asymptotic_order_interval"]
    for row in rows:
        if row["n"] == protocol["levels"][-1]:
            checks.append(dict(check="last_pair_order", case=row["case"], method=row["method"],
                               observed_order=row["observed_order"], passed=lower <= row["observed_order"] <= upper))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_files = ["studies/interpolation.py", "studies/plot_convergence.py", "studies/protocol.json", "src/lie_fem/groups.py"]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head, git_dirty = None, None
    report = dict(protocol=protocol, rows=rows, quadrature_verification=verification, checks=checks,
                  passed=all(check["passed"] for check in checks),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  provenance=dict(git_head=git_head, git_dirty=git_dirty, source_sha256={f: sha256(ROOT/f) for f in source_files}),
                  admissibility_counterexample=admissibility_counterexample())
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"Study checks failed; preserved all evidence in {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new run directory")
    args = parser.parse_args()
    run(args.output)
    from .plot_convergence import plot
    plot(args.output / "report.json", args.output / "plots")
    print(f"All checks passed. Raw fields, report, and plots: {args.output.resolve()}")


if __name__ == "__main__":
    main()
