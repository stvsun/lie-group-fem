"""Deterministic manufactured fields and non-overwriting example output."""

import argparse
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy

VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def rotation_field(points):
    angle = .15 + points @ [.3, -.2, .4]
    result = np.zeros((len(points), 3, 3))
    c, s = np.cos(angle), np.sin(angle)
    result[:, 0, 0] = result[:, 1, 1] = c
    result[:, 0, 1], result[:, 1, 0], result[:, 2, 2] = -s, s, 1.
    return result


def deformation_field(points):
    logarithm = points @ np.array([[.2, -.1, -.1], [-.1, .3, -.2], [.1, .1, -.2]])
    return rotation_field(points) * np.exp(logarithm)[:, None, :]


def save(name, metrics, **arrays):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    if args.output is None:
        Path("outputs").mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=name + "-", dir="outputs"))
    else:
        directory = args.output
        directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "fields.npz", **arrays)
    report = {
        "example": name,
        "units": "dimensionless tensors, radians, unit-length reference tetrahedra",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "metrics": metrics,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved raw fields and report: {directory.resolve()}")
