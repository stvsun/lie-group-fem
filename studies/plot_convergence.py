"""Plot a completed study report without rerunning or modifying its data."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(report_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(report_path).read_text())
    if not report["passed"]:
        raise ValueError("Cannot publish plots of a failed study as verified results")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 180,
    })
    styles = {"lie": ("#087e8b", "o", "Lie-group"), "componentwise": ("#c4511b", "s", "Componentwise")}

    def panel(axis, case, metric, title, ylabel, show_order=False):
        for method, (color, marker, label) in styles.items():
            rows = [r for r in report["rows"] if r["case"] == case and r["method"] == method]
            h = np.array([r["h"] for r in rows])
            values = np.array([r[metric] for r in rows])
            legend = label + (f"  (last p = {rows[-1]['observed_order']:.2f})" if show_order else "")
            axis.loglog(h, values, marker=marker, color=color, linewidth=2, markersize=6, label=legend)
        if show_order:
            reference = values[-1] * (h / h[-1])**2 * .55
            axis.loglog(h, reference, "--", color="#737b82", linewidth=1.3, label=r"$O(h^2)$ reference")
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Maximum tetrahedron diameter h")
        axis.set_ylabel(ylabel)
        axis.set_xticks(h, [f"{value:.3g}" for value in h])
        axis.minorticks_off()
        axis.invert_xaxis()
        axis.grid(True, which="major", color="#dfe4e8", linewidth=.7)
        axis.legend(loc="best", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    figure.suptitle("SO(3): nonlinear rotation field on refined tetrahedral meshes", fontsize=14)
    panel(axes[0], "so3", "relative_l2", "Matrix approximation error", "Relative matrix L² error", True)
    panel(axes[1], "so3", "orthogonality_rms", "Departure from the rotation group", "RMS ‖RᵀR − I‖F")
    figure.savefig(output / "so3-convergence.png")
    figure.savefig(output / "so3-convergence.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), layout="constrained")
    figure.suptitle("GL⁺(3): interpolation of tensor fields on refined tetrahedral meshes", fontsize=14)
    for i, (case, label) in enumerate((("gl3_compressible", "Compressible field"), ("gl3_isochoric", "Isochoric field (det F = 1)"))):
        panel(axes[i, 0], case, "relative_l2", label + ": matrix error", "Relative matrix L² error", True)
        panel(axes[i, 1], case, "relative_determinant_l2", label + ": volume error", "Relative L² error in det F")
    figure.savefig(output / "gl3-convergence.png")
    figure.savefig(output / "gl3-convergence.svg")
    plt.close(figure)
    (output / "render.json").write_text(json.dumps({"matplotlib": matplotlib.__version__, "source_report": str(Path(report_path).resolve())}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
