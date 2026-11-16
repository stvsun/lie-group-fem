"""Auditable replay of the specified analytic examples in Mota et al. (2013).

Run with --output NEW_DIRECTORY. Comparisons with printed paper numbers are
reported separately from analytic verification. No mechanics solver is invoked.
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.linalg import expm, logm

from lie_fem import MatrixChart, PositiveRealChart, interpolate
from .paper_hex import (BoxMesh, shape, project, ring_fields, ring_coordinates,
                        ring_decode, recovered_ring, error)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("paper_protocol.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_examples(output, tolerance):
    weights = np.array([[.5, .5], [-.5, 1.5]])
    sources = {
        "R+": np.array([[.9], [.1]]),
        "GL+": np.array([[[2, 0, 4], [0, 2, 0], [0, 0, 2]],
                         [[2, 0, 0], [0, 2, 0], [4, 0, 2]]]),
        "SL": np.array([[[1, 2, 0], [0, 1, 0], [0, 0, 1]],
                        [[1, 0, 0], [2, 1, 0], [0, 0, 1]]]),
        "SO": np.array([[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]]),
    }
    printed_lie = {
        "R+": [[.30], [.03]],
        "GL+": [[[3.09, 0, 2.35], [0, 2, 0], [2.35, 0, 3.09]],
                [[-.32, 0, -1.14], [0, 2, 0], [3.42, 0, -.32]]],
        "SL": [[[1.54, 1.18, 0], [1.18, 1.54, 0], [0, 0, 1]],
               [[-.16, -.57, 0], [1.71, -.16, 0], [0, 0, 1]]],
        "SO": [[[.72, .28, .63], [.28, .72, -.63], [-.63, .63, .44]],
               [[-.61, -.54, .58], [-.54, .82, .19], [-.58, -.19, -.79]]],
    }
    printed_direct = {
        "R+": [[.5], [-.3]],
        "GL+": [[[2, 0, 2], [0, 2, 0], [2, 0, 2]], [[2, 0, -2], [0, 2, 0], [6, 0, 2]]],
        "SL": [[[1, 1, 0], [1, 1, 0], [0, 0, 1]], [[1, -1, 0], [3, 1, 0], [0, 0, 1]]],
        "SO": [[[.5, 0, .5], [0, .5, -.5], [-.5, .5, 0]],
               [[-.5, 0, 1.5], [0, 1.5, .5], [-1.5, -1.5, 0]]],
    }
    records, raw = [], {"weights": weights}
    for name, source in sources.items():
        chart = PositiveRealChart() if name == "R+" else MatrixChart(name, 3)
        direct = np.einsum("qi,i...->q...", weights, source)
        lie = interpolate(source, weights, chart)
        for method, values, printed in (("componentwise", direct, printed_direct[name]),
                                        ("lie", lie, printed_lie[name])):
            discrepancy = float(np.max(np.abs(values - printed)))
            records.append({"group": name, "method": method,
                            "max_printed_difference": discrepancy,
                            "matches_printed_rounding": discrepancy <= tolerance,
                            "values": values.tolist()})
            raw[name + "_" + method] = values
        raw[name + "_source"] = source
    np.savez_compressed(output / "tables-2-3.npz", **raw)
    return records


def scalar_example(output, config):
    mesh = BoxMesh(config["counts"], config["lower"], config["upper"])
    points, weights, basis = mesh.quadrature(config["quadrature_order"])
    source = np.where(points[:, 0] < 0, points[:, 0] + 1, points[:, 0])[:, None]
    coefficients, residual = project(basis, weights, source)
    local_coordinates = (points[:8] - mesh.origins[0]) / mesh.spacing
    local_basis = shape(local_coordinates)
    local_coefficients = np.linalg.solve(local_basis, source.reshape(-1, 8, 1))
    oracle = mesh.nodes[:, 0] / 4 + .5
    local_reconstructed = np.einsum("qa,eak->eqk", local_basis, local_coefficients).reshape(-1, 1)
    np.savez_compressed(output / "scalar.npz", nodes=mesh.nodes, cells=mesh.cells,
                        points=points, weights=weights, source=source,
                        global_coefficients=coefficients, local_coefficients=local_coefficients,
                        global_values=basis @ coefficients, local_values=local_reconstructed)
    return {"global_nodal_oracle_max": float(np.max(np.abs(coefficients[:, 0] - oracle))),
            "local_sample_max": float(np.max(np.abs(local_reconstructed - source))),
            "global_L2": float(np.sqrt(np.dot(weights, (basis @ coefficients - source)[:, 0] ** 2))),
            "stationarity": residual}


def ring_study(output, config):
    mesh = BoxMesh(config["coarse_counts"], config["lower"], config["upper"])
    fine = BoxMesh(config["fine_counts"], config["lower"], config["upper"])
    target_basis = mesh.basis(fine.nodes)
    target_exact = ring_fields(fine.nodes)[0]
    raw = {"source_nodes": mesh.nodes, "source_cells": mesh.cells,
           "source_exact": ring_fields(mesh.nodes)[0], "target_nodes": fine.nodes,
           "target_cells": fine.cells, "target_exact": target_exact}
    interpolation = []
    for method in ("direct", "mixed", "lie", "matrix_log"):
        if method == "matrix_log":
            # Preserve complex logs and exponentials instead of silently taking real parts.
            logs = np.array([logm(f.astype(complex)) for f in raw["source_exact"]])
            evaluated_logs = (target_basis @ logs.reshape(-1, 9)).reshape(-1, 3, 3)
            approximate = np.array([expm(value) for value in evaluated_logs])
            raw["matrix_log_source_coordinates"] = logs
        else:
            coordinates = ring_coordinates(mesh.nodes, method)
            approximate = ring_decode(target_basis @ coordinates, method)[0]
            raw[method + "_source_coordinates"] = coordinates
        raw[method + "_target"] = approximate
        maximum = float(np.linalg.norm(approximate - target_exact, axis=(1, 2)).max())
        published = config["published_interpolation_max"][method]
        tolerance = 1e-11 if published < 1e-10 else (0.05 if method == "matrix_log" else 5.01e-5)
        interpolation.append({"method": method, "sampled_max": maximum, "published_max": published,
                              "max_imaginary": float(np.max(np.abs(approximate.imag))),
                              "matches_published": abs(maximum - published) <= tolerance})
    np.savez_compressed(output / "ring-interpolation.npz", **raw)

    recovery = []
    qp, qw, qb = mesh.quadrature(config["quadrature_order"])
    ap, aw, ab = mesh.quadrature(config["audit_quadrature_order"])
    raw = {"source_nodes": mesh.nodes, "source_cells": mesh.cells, "source_points": qp,
           "source_weights": qw, "source_exact": ring_fields(qp)[0],
           "audit_points": ap, "audit_weights": aw, "audit_exact": ring_fields(ap)[0]}
    for method in ("nearest", "extrapolation", "direct", "direct_F", "mixed", "lie"):
        coefficients, residual, decoder = recovered_ring(mesh, method)
        nodal = ring_decode(coefficients, decoder)[0]
        at_source = ring_decode(qb @ coefficients, decoder)[0]
        at_audit = ring_decode(ab @ coefficients, decoder)[0]
        point_errors = np.linalg.norm(nodal - ring_fields(mesh.nodes)[0], axis=(1, 2))
        current = [float(point_errors.min()), float(point_errors.max()), error(at_source, raw["source_exact"], qw)]
        published = config["published_recovery"][method]
        # Three significant figures in Table 7; roundoff entries compared as such.
        matches = [bool(abs(a - b) <= max(1e-11, .0051 * 10 ** np.floor(np.log10(b))))
                   for a, b in zip(current, published)]
        recovery.append({"method": method, "nodal_min": current[0], "nodal_max": current[1],
                         "source_quadrature_L2": current[2],
                         "independent_quadrature_L2": error(at_audit, raw["audit_exact"], aw),
                         "published": published, "matches_published": matches,
                         "stationarity": residual})
        raw[method + "_coefficients"] = coefficients
        raw[method + "_nodal"] = nodal
        raw[method + "_source_evaluated"] = at_source
        raw[method + "_audit_evaluated"] = at_audit
    np.savez_compressed(output / "ring-recovery.npz", **raw)

    convergence = []
    for factor in config["refinement_factors"]:
        refined = BoxMesh([4 * factor, factor, 1], config["lower"], config["upper"])
        raw = {"nodes": refined.nodes, "cells": refined.cells}
        for order in (config["quadrature_order"], config["audit_quadrature_order"]):
            p, w, b = refined.quadrature(order)
            exact = ring_fields(p)
            raw[f"q{order}_points"], raw[f"q{order}_weights"] = p, w
            raw[f"q{order}_exact"] = exact[0]
            for method in ("direct", "mixed", "lie"):
                coefficients = ring_coordinates(refined.nodes, method)
                approximate = ring_decode(b @ coefficients, method)
                raw[f"{method}_coefficients"] = coefficients
                raw[f"q{order}_{method}"] = approximate[0]
                convergence.append({"elements": len(refined.cells), "h_x": float(refined.spacing[0]),
                                    "quadrature_order": order, "method": method,
                                    **{f"E_{name}": error(a, e, w) for name, a, e in zip("FRS", approximate, exact)}})
        np.savez_compressed(output / f"ring-convergence-{factor}.npz", **raw)
    return interpolation, recovery, convergence


def verify(tables, scalar, interpolation, recovery, convergence, tolerance):
    checks = {"table3_printed_rounding": all(r["matches_printed_rounding"] for r in tables if r["method"] == "lie"),
              "scalar_global_affine_oracle": scalar["global_nodal_oracle_max"] < tolerance,
              "scalar_local_exact": scalar["local_sample_max"] < tolerance,
              "scalar_L2_oracle": abs(scalar["global_L2"] - np.sqrt(.5)) < tolerance,
              "ring_mixed_interpolation_exact": next(r for r in interpolation if r["method"] == "mixed")["sampled_max"] < tolerance,
              "ring_mixed_recovery_exact": next(r for r in recovery if r["method"] == "mixed")["nodal_max"] < tolerance,
              "ring_mixed_independent_error": next(r for r in recovery if r["method"] == "mixed")["independent_quadrature_L2"] < tolerance,
              "projection_stationarity": max(r["stationarity"] for r in recovery) < tolerance}
    for method in ("direct", "mixed", "lie"):
        rows = [r for r in convergence if r["method"] == method and r["quadrature_order"] == 5]
        if method == "mixed":
            checks["mixed_all_levels_exact"] = max(r["E_F"] for r in rows) < tolerance
        else:
            rate = np.log2(rows[-2]["E_F"] / rows[-1]["E_F"])
            checks[f"{method}_second_order_h"] = bool(1.9 < rate < 2.1)
        exact_factor = "E_R" if method == "lie" else "E_S"
        checks[f"{method}_exact_factor"] = max(r[exact_factor] for r in rows) < tolerance
    return {name: bool(passed) for name, passed in checks.items()}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    tables = table_examples(output, protocol["verification"]["printed_table_tolerance"])
    scalar = scalar_example(output, protocol["scalar"])
    interpolation, recovery, convergence = ring_study(output, protocol["ring"])
    checks = verify(tables, scalar, interpolation, recovery, convergence,
                    protocol["verification"]["analytic_tolerance"])
    report = {"schema": 1, "scope": protocol["scope"], "assumptions": protocol["ring"]["assumptions"],
              "exact_published_replication": False,
              "status": "Analytic replay verified separately; printed discrepancies retained.",
              "tables": tables, "scalar": scalar, "ring_interpolation": interpolation,
              "ring_recovery": recovery, "convergence": convergence, "checks": checks,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": version("matplotlib")},
              "source_hashes": {name: digest(ROOT / name) for name in [
                  "src/lie_fem/classical.py", "src/lie_fem/groups.py", "studies/paper_hex.py",
                  "studies/paper_examples.py", "studies/paper_plots.py", "studies/paper_protocol.json"]}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "ring-convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(convergence[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(convergence)
    from .paper_plots import plot
    plot(output)
    (output / "manifest.json").write_text(json.dumps({
        path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "checks": checks,
                      "published_replication": report["exact_published_replication"]}, indent=2))
    if not all(checks.values()):
        raise SystemExit("Analytic verification failed; raw evidence retained")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.output)
