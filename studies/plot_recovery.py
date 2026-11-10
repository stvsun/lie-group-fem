"""Render a verified recovery/transfer report to independent PNG/SVG figures."""

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
        raise ValueError("Cannot present failed study checks as verified plots")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 180})
    styles = {"lie": ("#087e8b", "o", "Lie-group"), "componentwise": ("#c4511b", "s", "Componentwise")}
    cases = [("so3", "SO(3)"), ("gl3_compressible", "GL⁺(3), compressible"),
             ("gl3_isochoric", "GL⁺(3), isochoric")]
    for stage, heading in (("recovery", "Variational recovery from source integration points"),
                           ("transfer", "Full transfer: source recovery → target nodes → target FE field")):
        figure, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
        figure.suptitle(heading, fontsize=14)
        for row, (case, label) in enumerate(cases):
            for column, metric in enumerate(("relative_l2", "orthogonality_rms" if case == "so3" else "relative_determinant_l2")):
                axis = axes[row, column]
                for method, (color, marker, name) in styles.items():
                    data = [r for r in report["rows"] if r["case"] == case and r["stage"] == stage and r["method"] == method]
                    h = np.array([r["h"] for r in data])
                    values = np.array([r[metric] for r in data])
                    legend = name + (f" (last p = {data[-1]['observed_order']:.2f})" if column == 0 else "")
                    axis.loglog(h, values, marker=marker, color=color, linewidth=2, markersize=5, label=legend)
                if column == 0:
                    axis.loglog(h, .55*values[-1]*(h/h[-1])**2, "--", color="#737b82", label=r"$O(h^2)$ reference")
                axis.set_title(label + (": matrix error" if column == 0 else ": orthogonality" if case == "so3" else ": determinant error"), loc="left", fontweight="bold")
                axis.set_ylabel("Relative matrix L² error" if column == 0 else "RMS ‖RᵀR − I‖F" if case == "so3" else "Relative L² error in det F")
                axis.set_xlabel("Maximum source tetrahedron diameter h")
                axis.set_xticks(h, [f"{x:.3g}" for x in h])
                axis.minorticks_off()
                axis.invert_xaxis()
                axis.grid(True, color="#dfe4e8", linewidth=.7)
                axis.legend(fontsize=8.5, loc="best", frameon=True, edgecolor="#e0e0e0")
        figure.savefig(output/f"{stage}-convergence.png")
        figure.savefig(output/f"{stage}-convergence.svg")
        plt.close(figure)
    (output/"render.json").write_text(json.dumps({"matplotlib": matplotlib.__version__, "source_report": str(Path(report_path).resolve())}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
