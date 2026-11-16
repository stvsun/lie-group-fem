"""All Table 1 group families: affine-chart recovery on a tetrahedron.

Examples are illustrative library checks, not additional paper benchmarks.
"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from lie_fem import (MatrixChart, PositiveRealChart, recover,
                     tetrahedron_quadrature, tetrahedron_shape)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    points, measures = tetrahedron_quadrature(vertices)
    basis = tetrahedron_shape(vertices, points)
    target_points = np.array([[.25, .25, .25], [.1, .2, .3]])
    target_basis = tetrahedron_shape(vertices, target_points)
    charts = {"R+": PositiveRealChart(),
              "GL_negative": MatrixChart("GL", 3, np.diag([-1., 1, 1])),
              "GL+": MatrixChart("GL+", 3), "SL": MatrixChart("SL", 3),
              "O_negative": MatrixChart("O", 3, np.diag([-1., 1, 1])),
              "SO": MatrixChart("SO", 3), "Sp": MatrixChart("Sp", 2),
              "SE": MatrixChart("SE", 3)}
    raw = {"vertices": vertices, "source_points": points, "measures": measures,
           "target_points": target_points, "source_basis": basis, "target_basis": target_basis}
    rows = []
    for name, chart in charts.items():
        coefficients = .1 * np.sin(np.arange(4 * chart.dimension).reshape(4, -1))
        values = chart.decode(basis @ coefficients)
        field = recover(values, basis, measures, chart)
        transferred = field.evaluate(target_basis)
        expected = chart.decode(target_basis @ coefficients)
        discrepancy = float(np.max(np.abs(transferred - expected)))
        rows.append({"group": name, "coordinates": chart.dimension,
                     "max_transfer_error": discrepancy, "stationarity": field.stationarity_residual})
        raw[name + "_source"] = values
        raw[name + "_exact_coefficients"] = coefficients
        raw[name + "_recovered_coefficients"] = field.coordinates
        raw[name + "_target"] = transferred
        raw[name + "_target_exact"] = expected
    np.savez_compressed(output / "fields.npz", **raw)
    report = {"examples": rows, "versions": {"python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__},
              "units": "Dimensionless tensor/rotation coordinates; SE translations and mesh coordinates use one consistent length unit.",
              "passed": all(row["max_transfer_error"] < 2e-11 for row in rows)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Group mapping check failed; raw evidence preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
