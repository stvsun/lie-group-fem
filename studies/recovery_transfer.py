"""Global variational recovery and transfer to a nonmatching target FE mesh.

This study-only sparse assembly implements the same chart-space L2 objective
as lie_fem.recover. Small meshes are checked against that public dense SVD API.
Source sampling, recovery, target-node mapping and target evaluation are explicit.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .interpolation import ROOT, METHODS, chart_for, cube_mesh, manufactured, sha256, tetrahedron_rule


def geometry(vertices, cells):
    elements = vertices[cells]
    determinants = np.linalg.det((elements[:, 1:] - elements[:, :1]).transpose(0, 2, 1))
    if np.any(determinants <= 0) or not np.all(np.isfinite(determinants)):
        raise ValueError("invalid mesh orientation or volume")
    diameter = max(float(np.linalg.norm(elements[:, a]-elements[:, b], axis=1).max())
                   for a, b in itertools.combinations(range(4), 2))
    return determinants, diameter


def target_mesh(n):
    """Different triangulation and interior node locations, same cube boundary."""
    vertices, cells = cube_mesh(n)
    vertices[:, 0] = 1 - vertices[:, 0]
    cells = cells[:, [0, 2, 1, 3]]  # Reflection reverses orientation.
    bump = .15/n*np.prod(np.sin(np.pi*vertices), axis=1)
    vertices += bump[:, None]*np.array([1., -.6, .4])
    geometry(vertices, cells)
    return vertices, cells


def locate_source(n, points):
    """Exact structured Freudenthal locator; returns four node IDs and weights.

    Only this study's source grid is supported. Stable ordering gives a
    deterministic trace on shared faces. Outside points are rejected.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite (n_points,3) coordinates")
    if np.any(points < -1e-12) or np.any(points > 1+1e-12):
        raise ValueError("point is outside the source domain")
    scaled = np.clip(points, 0., 1.)*n
    origin = np.minimum(np.floor(scaled).astype(int), n-1)
    fractional = scaled-origin
    permutation = np.argsort(-fractional, axis=1, kind="stable")
    sorted_fractional = np.take_along_axis(fractional, permutation, axis=1)
    first = origin + np.eye(3, dtype=int)[permutation[:, 0]]
    second = first + np.eye(3, dtype=int)[permutation[:, 1]]
    corners = np.stack([origin, first, second, origin+1], axis=1)
    nodes = corners @ np.array([(n+1)**2, n+1, 1])
    a, b, c = sorted_fractional.T
    weights = np.column_stack([1-a, a-b, b-c, c])
    return nodes, weights


def evaluate_coordinates(coefficients, nodes, shape):
    return np.einsum("qa,qak->qk", shape, coefficients[nodes])


def consistent_mass(vertices, cells):
    """Exact P1 consistent mass: V/20*(ones(4,4)+eye(4))."""
    determinants, _ = geometry(vertices, cells)
    local = determinants[:, None, None]/120*(np.ones((4, 4))+np.eye(4))
    row = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    column = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    return coo_matrix((local.ravel(), (row, column)), shape=(len(vertices), len(vertices))).tocsc()


def fit(n, case, order, directory=None, field=manufactured, batch_elements=256):
    """Recover both methods from identical source integration-point matrices."""
    vertices, cells = cube_mesh(n)
    determinants, h = geometry(vertices, cells)
    shape, reference_weights = tetrahedron_rule(order)
    chart = chart_for(case)
    dimension = 3 if case == "so3" else 9
    rhs = np.zeros((len(vertices), dimension+9))
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/"source-mesh.npz", vertices=vertices, cells=cells,
                            quadrature_shape=shape, reference_weights=reference_weights)
    for start in range(0, len(cells), batch_elements):
        batch = cells[start:start+batch_elements]
        points = np.einsum("qa,eai->eqi", shape, vertices[batch]).reshape(-1, 3)
        source = field(points, case)
        encoded = chart.encode(source)
        data = np.column_stack([encoded, source.reshape(-1, 9)])
        data = data.reshape(len(batch), len(shape), dimension+9)
        weights = determinants[start:start+len(batch), None]*reference_weights
        local_rhs = np.einsum("qa,eq,eqk->eak", shape, weights, data)
        np.add.at(rhs, batch.ravel(), local_rhs.reshape(-1, dimension+9))
        if directory is not None:
            np.savez_compressed(directory/f"source-{start:06d}.npz", points=points,
                                measures=weights.ravel(), values=source, chart_coordinates=encoded,
                                elements=np.arange(start, start+len(batch)))
    mass = consistent_mass(vertices, cells)
    # All generated meshes are conforming P1 meshes with positive volumes:
    # their consistent mass matrix is SPD. No regularization or lumping.
    coefficients = splu(mass).solve(rhs)
    lie, ordinary = coefficients[:, :dimension], coefficients[:, dimension:]
    chart.decode(lie)  # Fail if extrapolated nodal values leave the local chart.
    residual = mass@coefficients-rhs
    diagnostics = {}
    for method, indices in (("lie", slice(0, dimension)), ("componentwise", slice(dimension, None))):
        diagnostics[method] = {
            "relative_normal_residual": float(np.linalg.norm(residual[:, indices])/np.linalg.norm(rhs[:, indices])),
        }
    if directory is not None:
        np.savez_compressed(directory/"recovered.npz", lie_coordinates=lie,
                            componentwise_coefficients=ordinary, right_hand_side=rhs,
                            mass_data=mass.data, mass_indices=mass.indices, mass_indptr=mass.indptr,
                            mass_shape=mass.shape, reference=chart.reference)
    return dict(n=n, case=case, vertices=vertices, cells=cells, lie=lie, componentwise=ordinary,
                chart=chart, source_order=order, source_h=h, diagnostics=diagnostics)


def evaluate_stage(fitted, stage, order, directory=None, batch_elements=256):
    """Compare to analytic truth at independent integration points.

    transfer: recovered source field -> target nodes -> target FE quadrature.
    The direct source-field evaluation at target quadrature is also retained
    to distinguish transfer representation error from source recovery error.
    """
    n, case, chart = fitted["n"], fitted["case"], fitted["chart"]
    if stage == "recovery":
        vertices, cells = fitted["vertices"], fitted["cells"]
        nodal = {method: fitted[method] for method in METHODS}
        mapped_nodes = mapped_weights = None
    elif stage == "transfer":
        vertices, cells = target_mesh(n)
        mapped_nodes, mapped_weights = locate_source(n, vertices)
        nodal = {method: evaluate_coordinates(fitted[method], mapped_nodes, mapped_weights) for method in METHODS}
        chart.decode(nodal["lie"])
    else:
        raise ValueError("unknown stage")
    determinants, h_target = geometry(vertices, cells)
    shape, reference_weights = tetrahedron_rule(order)
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = dict(vertices=vertices, cells=cells, lie_coordinates=nodal["lie"],
                      componentwise_coefficients=nodal["componentwise"], quadrature_shape=shape,
                      reference_weights=reference_weights)
        if stage == "transfer":
            arrays.update(source_nodes=mapped_nodes, source_weights=mapped_weights)
        np.savez_compressed(directory/"evaluation-mesh.npz", **arrays)
    stats = {method: dict(error2=0., norm2=0., det_error2=0., det_norm2=0., orthogonality2=0.,
                         max_error=0., max_group_defect=0., min_det=float("inf"), nonpositive=0,
                         transfer2=0., direct_error2=0.) for method in METHODS}
    volume = 0.
    for start in range(0, len(cells), batch_elements):
        batch = cells[start:start+batch_elements]
        points = np.einsum("qa,eai->eqi", shape, vertices[batch]).reshape(-1, 3)
        weights = (determinants[start:start+len(batch), None]*reference_weights).ravel()
        truth = manufactured(points, case)
        truth_det = np.linalg.det(truth)
        approximate = {}
        for method in METHODS:
            coordinates = np.einsum("qa,eak->eqk", shape, nodal[method][batch]).reshape(-1, nodal[method].shape[1])
            approximate[method] = chart.decode(coordinates) if method == "lie" else coordinates.reshape(-1, 3, 3)
        direct = approximate
        source_nodes = source_weights = None
        if stage == "transfer":
            source_nodes, source_weights = locate_source(n, points)
            direct = {}
            for method in METHODS:
                coordinates = evaluate_coordinates(fitted[method], source_nodes, source_weights)
                direct[method] = chart.decode(coordinates) if method == "lie" else coordinates.reshape(-1, 3, 3)
        volume += float(weights.sum())
        for method, value in approximate.items():
            s = stats[method]
            error2 = np.sum((value-truth)**2, axis=(1, 2))
            determinant = np.linalg.det(value)
            defect2 = np.sum((value.transpose(0, 2, 1)@value-np.eye(3))**2, axis=(1, 2))
            s["error2"] += float(weights@error2)
            s["norm2"] += float(weights@np.sum(truth**2, axis=(1, 2)))
            s["det_error2"] += float(weights@(determinant-truth_det)**2)
            s["det_norm2"] += float(weights@truth_det**2)
            s["orthogonality2"] += float(weights@defect2)
            s["max_error"] = max(s["max_error"], float(np.sqrt(error2.max())))
            s["max_group_defect"] = max(s["max_group_defect"], float(np.sqrt(defect2.max())) if case == "so3" else float(np.abs(determinant-1).max()))
            s["min_det"] = min(s["min_det"], float(determinant.min()))
            s["nonpositive"] += int(np.count_nonzero(determinant <= 0))
            s["transfer2"] += float(weights@np.sum((value-direct[method])**2, axis=(1, 2)))
            s["direct_error2"] += float(weights@np.sum((direct[method]-truth)**2, axis=(1, 2)))
        if directory is not None:
            arrays = dict(points=points, measures=weights, exact=truth, **approximate,
                          elements=np.arange(start, start+len(batch)))
            if stage == "transfer":
                arrays.update(source_nodes=source_nodes, source_weights=source_weights,
                              lie_direct=direct["lie"], componentwise_direct=direct["componentwise"])
            np.savez_compressed(directory/f"evaluation-{start:06d}.npz", **arrays)
    result = []
    for method, s in stats.items():
        result.append(dict(case=case, stage=stage, method=method, n=n, h=fitted["source_h"],
                           evaluation_h=h_target, source_nodes=len(fitted["vertices"]), target_nodes=len(vertices),
                           elements=len(cells), samples=len(cells)*len(shape), volume=volume,
                           source_order=fitted["source_order"], evaluation_order=order,
                           relative_l2=float(np.sqrt(s["error2"]/s["norm2"])),
                           sampled_max_error=s["max_error"],
                           relative_determinant_l2=float(np.sqrt(s["det_error2"]/s["det_norm2"])),
                           orthogonality_rms=float(np.sqrt(s["orthogonality2"]/volume)) if case == "so3" else None,
                           sampled_max_group_defect=s["max_group_defect"] if case != "gl3_compressible" else None,
                           sampled_min_determinant=s["min_det"], nonpositive_samples=s["nonpositive"],
                           sampled_transfer_increment=float(np.sqrt(s["transfer2"]/s["norm2"])),
                           sampled_direct_source_error=float(np.sqrt(s["direct_error2"]/s["norm2"])),
                           **fitted["diagnostics"][method]))
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_bytes = (ROOT/"studies/recovery_protocol.json").read_bytes()
    (output/"protocol.json").write_bytes(protocol_bytes)
    protocol = json.loads(protocol_bytes)
    rows, verification, checks = [], [], []
    options = dict(batch_elements=protocol["raw_batch_elements"])
    for case in protocol["cases"]:
        for n in protocol["levels"]:
            directory = output/"raw"/case/f"n{n:03d}"
            fitted = fit(n, case, protocol["source_quadrature_order"], directory/"source", **options)
            baseline = []
            for stage in protocol["stages"]:
                current = evaluate_stage(fitted, stage, protocol["evaluation_quadrature_order"], directory/stage, **options)
                rows.extend(current)
                baseline.extend(current)
                print(f"{case}, {stage}, n={n}: " + ", ".join(f"{r['method']} L2={r['relative_l2']:.6e}" for r in current), flush=True)
            if n == protocol["levels"][-1]:
                refined = fit(n, case, protocol["verification_source_order"], directory/"verify-source", **options)
                for kind in ("source_quadrature", "evaluation_quadrature"):
                    for stage in protocol["stages"]:
                        field = refined if kind == "source_quadrature" else fitted
                        order = protocol["evaluation_quadrature_order"] if kind == "source_quadrature" else protocol["verification_evaluation_order"]
                        current = evaluate_stage(field, stage, order, directory/f"verify-{kind}-{stage}", **options)
                        verification.extend(dict(check_kind=kind, **row) for row in current)
                        base = [row for row in baseline if row["stage"] == stage]
                        for a, b in zip(base, current):
                            for metric in ("relative_l2", "relative_determinant_l2"):
                                relative = abs(a[metric]-b[metric])/max(b[metric], protocol["roundoff_threshold"])
                                tiny = max(a[metric], b[metric]) < protocol["roundoff_threshold"]
                                checks.append(dict(check=kind, case=case, stage=stage, method=a["method"], metric=metric,
                                                   relative_change=relative, roundoff_scale=tiny,
                                                   passed=tiny or relative <= protocol["quadrature_relative_tolerance"]))
    previous = {}
    for row in rows:
        key = row["case"], row["stage"], row["method"]
        old = previous.get(key)
        row["observed_order"] = None if old is None else float(np.log(old["relative_l2"]/row["relative_l2"])/np.log(old["h"]/row["h"]))
        previous[key] = row
        checks.append(dict(check="normal_equation", case=row["case"], stage=row["stage"], method=row["method"], n=row["n"],
                           residual=row["relative_normal_residual"], passed=row["relative_normal_residual"] < protocol["normal_equation_relative_tolerance"]))
        if row["method"] == "lie":
            metric = row["sampled_max_group_defect"]
            checks.append(dict(check="group_membership", case=row["case"], stage=row["stage"], n=row["n"],
                               sampled_max_defect=metric, sampled_min_determinant=row["sampled_min_determinant"],
                               passed=row["sampled_min_determinant"] > 0 and (metric is None or metric < protocol["group_defect_tolerance"])))
        if row["n"] == protocol["levels"][-1]:
            lower, upper = protocol["last_pair_order_interval"]
            checks.append(dict(check="last_pair_order", case=row["case"], stage=row["stage"], method=row["method"],
                               order=row["observed_order"], passed=lower <= row["observed_order"] <= upper))
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_files = ["studies/recovery_transfer.py", "studies/plot_recovery.py", "studies/recovery_protocol.json",
                    "studies/interpolation.py", "src/lie_fem/groups.py", "src/lie_fem/fem.py"]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        head, dirty = None, None
    report = dict(protocol=protocol, rows=rows, verification=verification, checks=checks,
                  passed=all(c["passed"] for c in checks),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  provenance=dict(git_head=head, git_dirty=dirty, source_sha256={f: sha256(ROOT/f) for f in source_files}))
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    (output/"sha256.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not report["passed"]:
        failed = [c for c in checks if not c["passed"]]
        print(json.dumps(failed, indent=2), flush=True)
        raise RuntimeError(f"Study checks failed; all evidence retained in {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
    from .plot_recovery import plot
    plot(args.output/"report.json", args.output/"plots")
    print(f"All checks passed. Evidence and plots: {args.output.resolve()}")


if __name__ == "__main__":
    main()
