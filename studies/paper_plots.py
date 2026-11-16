"""Publication-style plots from preserved analytic benchmark output."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.ticker import NullFormatter
import numpy as np

COLORS = {"direct": "#b64c32", "mixed": "#276baf", "lie": "#278565"}
LABELS = {"direct": "Direct R, direct S", "mixed": "Lie R, direct S", "lie": "Lie R, Lie S"}


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def plot(output):
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "lie-fem-paper-2013"})
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for a, b in ((-1, 0), (0, 1)):
        x = np.linspace(a, b, 200)
        z = x + 1 if a < 0 else x
        ax.plot(x, z, color="#278565", linewidth=3, label="Exact / local recovery" if a < 0 else None)
    x = np.linspace(-1, 1, 300)
    ax.plot(x, x / 4 + .5, color="#b64c32", label="Global consistent L² recovery")
    ax.set(xlabel="X", ylabel="Scalar field", title="Section 6.1 · 8 Q1 hexahedra, 8 Gauss points per element")
    ax.legend(loc="upper left")
    ax.grid(alpha=.2)
    save(fig, output, "paper-scalar")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    for ax, field in zip(axes, "FRS"):
        for method, color in COLORS.items():
            rows = [r for r in report["convergence"] if r["method"] == method and r["quadrature_order"] == 5]
            ax.loglog([r["h_x"] for r in rows], [max(r[f"E_{field}"], 1e-18) for r in rows],
                      "o-", color=color, label=LABELS[method], markersize=4)
        ax.set(xlabel="Axial element size hₓ", ylabel=f"Absolute volume L² error in {field}")
        ax.set_xticks([1 / 64, 1 / 16, 1 / 4], ["1/64", "1/16", "1/4"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="both", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Section 6.2.1 · L = 1, (4k, k, 1) Q1 meshes · independent 5³-point quadrature")
    save(fig, output, "paper-ring-convergence")

    raw = np.load(output / "ring-interpolation.npz")
    points = raw["target_nodes"]
    zmask = np.isclose(points[:, 2], points[:, 2].min())
    x = np.unique(points[:, 0]); y = np.unique(points[:, 1])
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), layout="constrained")
    for ax, row in zip(axes.flat, report["ring_interpolation"]):
        method = row["method"]
        values = np.linalg.norm(raw[method + "_target"] - raw["target_exact"], axis=(1, 2))
        grid = values[zmask].reshape(len(x), len(y)).T
        im = ax.pcolormesh(x, y, grid, shading="auto", cmap="magma", rasterized=True)
        fig.colorbar(im, ax=ax, label="Frobenius error")
        ax.set(xlabel="X / L", ylabel="Y / L", title=f"{LABELS.get(method, 'Complex matrix log F')} · max {row['sampled_max']:.4g}")
    fig.suptitle("Section 6.2 · analytic replay with declared 8×2×1 → 100×25×1 meshes")
    save(fig, output, "paper-ring-interpolation")

    # The exact deformed mid-surface: (x,y)=((R-Y)sin(X/R),-(R-Y)cos(X/R)).
    # Explicit triangles retain the original topology at the coincident beam ends.
    surface = points[zmask]
    radius = 1 / (2 * np.pi) - surface[:, 1]
    theta = 2 * np.pi * surface[:, 0]
    triangles = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            a = i * len(y) + j
            triangles.extend([[a, a + len(y), a + len(y) + 1], [a, a + len(y) + 1, a + 1]])
    triangulation = Triangulation(radius * np.sin(theta), -radius * np.cos(theta), triangles)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), layout="constrained")
    for ax, row in zip(axes.flat, report["ring_interpolation"]):
        method = row["method"]
        values = np.linalg.norm(raw[method + "_target"] - raw["target_exact"], axis=(1, 2))
        im = ax.tripcolor(triangulation, values[zmask], shading="gouraud", cmap="magma", rasterized=True)
        fig.colorbar(im, ax=ax, label="Frobenius error", shrink=.8)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{LABELS.get(method, 'Complex matrix log F')}\nmax {row['sampled_max']:.4g}")
    fig.suptitle("Bending into a ring · same analytic deformation, declared mesh and log branches")
    save(fig, output, "paper-ring-deformed")

    rows = report["ring_recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = ["Nearest", "Extrap.", "Polar\ndirect", "Direct F", "Polar\nmixed", "Polar\nLie"]
    positions = np.arange(len(rows))
    axes[0].semilogy(positions, [max(r["nodal_max"], 1e-18) for r in rows], "o", label="Replay")
    axes[0].semilogy(positions, [r["published"][1] for r in rows], "x", label="Printed Table 7")
    axes[0].set(ylabel="Maximum nodal Frobenius error")
    for key, marker, label in (("source_quadrature_L2", "o", "Replay · source 2³ quadrature"),
                               ("independent_quadrature_L2", "s", "Replay · independent 5³ quadrature")):
        axes[1].semilogy(positions, [max(r[key], 1e-18) for r in rows], marker, label=label)
    axes[1].semilogy(positions, [r["published"][2] for r in rows], "x", label="Printed Table 7")
    axes[1].set(ylabel="Absolute volume L² error (L = 1)")
    for ax in axes:
        ax.set_xticks(positions, names)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Section 6.2.2 · recovery comparisons · published discrepancies are retained")
    save(fig, output, "paper-ring-recovery")
