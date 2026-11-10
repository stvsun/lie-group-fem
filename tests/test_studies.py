import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from lie_fem import interpolate
from studies.interpolation import (
    admissibility_counterexample, chart_for, cube_mesh, level, manufactured,
    sample_interpolants, tetrahedron_rule,
)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_conforming_mesh_geometry(n):
    vertices, cells = cube_mesh(n)
    tetrahedra = vertices[cells]
    jac = np.linalg.det((tetrahedra[:, 1:] - tetrahedra[:, :1]).transpose(0, 2, 1))
    assert len(cells) == 6*n**3
    assert np.all(jac > 0)
    assert_allclose(jac.sum()/6, 1., atol=1e-14)
    lengths = [np.linalg.norm(tetrahedra[:, i] - tetrahedra[:, j], axis=1) for i, j in itertools.combinations(range(4), 2)]
    assert_allclose(np.max(lengths), np.sqrt(3)/n, atol=1e-14)
    faces = {}
    for cell in cells:
        for face in itertools.combinations(cell, 3):
            key = tuple(sorted(face))
            faces[key] = faces.get(key, 0) + 1
    for face, count in faces.items():
        assert count in (1, 2)
        if count == 1:
            coordinates = vertices[list(face)]
            assert any(np.all(coordinates[:, axis] == boundary) for axis in range(3) for boundary in (0, 1))


@pytest.mark.parametrize("order,degree", [(4, 5), (5, 7)])
def test_duffy_rule_against_exact_simplex_moments(order, degree):
    shape, weights = tetrahedron_rule(order)
    assert np.all(shape >= 0) and np.all(weights > 0)
    assert_allclose(shape.sum(axis=1), 1., atol=1e-15)
    x, y, z = shape[:, 1:].T
    for p, q, r in itertools.product(range(degree+1), repeat=3):
        if p + q + r <= degree:
            exact = math.factorial(p)*math.factorial(q)*math.factorial(r)/math.factorial(p+q+r+3)
            assert_allclose(weights @ (x**p*y**q*z**r), exact, rtol=2e-13, atol=1e-17)


@pytest.mark.parametrize("case", ["so3", "gl3_compressible", "gl3_isochoric"])
def test_manufactured_fields_and_batch_match_public_interpolation(case):
    vertices, cells = cube_mesh(1)
    shape, _ = tetrahedron_rule(4)
    exact = manufactured(vertices, case)
    chart = chart_for(case)
    coordinates = chart.encode(exact)
    points, lie, componentwise = sample_interpolants(exact, coordinates, cells, vertices, shape, chart)
    expected = np.concatenate([interpolate(exact[cell], shape, chart) for cell in cells])
    assert_allclose(lie, expected, atol=1e-14)
    ordinary = np.concatenate([np.tensordot(shape, exact[cell], axes=1) for cell in cells])
    assert_allclose(componentwise, ordinary, atol=1e-14)
    assert np.all(np.linalg.det(manufactured(points, case)) > 0)
    if case in ("so3", "gl3_isochoric"):
        assert_allclose(np.linalg.det(manufactured(points, case)), 1., atol=3e-14)
    else:
        x, y, z = points.T
        log_j = .25*np.sin(np.pi*x/2)+.12*y*z-.18*np.cos(np.pi*y/2)+.10*x*z+.10*x-.08*y+.12*np.sin(np.pi*z)
        assert_allclose(np.linalg.det(manufactured(points, case)), np.exp(log_j), atol=3e-14)


@pytest.mark.parametrize("case", ["so3", "gl3_compressible", "gl3_isochoric"])
def test_refinement_reduces_both_errors(case):
    coarse, fine = level(case, 1, 4), level(case, 2, 4)
    for a, b in zip(coarse, fine):
        assert b["relative_l2"] < a["relative_l2"]
        assert_allclose(b["volume"], 1., atol=1e-14)
    if case == "so3":
        assert fine[0]["orthogonality_rms"] < 3e-14
        assert fine[1]["orthogonality_rms"] > 1e-4
    if case == "gl3_isochoric":
        assert fine[0]["relative_determinant_l2"] < 3e-14
        assert fine[1]["relative_determinant_l2"] > 1e-4


def test_saved_raw_arrays_reproduce_metrics_and_are_not_overwritten(tmp_path):
    directory = tmp_path / "raw"
    rows = level("so3", 1, 4, directory, batch_elements=2)
    sources = list(directory.glob("samples-*.npz"))
    error, norm = 0., 0.
    for path in sources:
        with np.load(path, allow_pickle=False) as data:
            error += data["measures"] @ np.sum((data["lie"]-data["exact"])**2, axis=(1, 2))
            norm += data["measures"] @ np.sum(data["exact"]**2, axis=(1, 2))
    assert_allclose(rows[0]["relative_l2"], np.sqrt(error/norm), atol=1e-16)
    before = sources[0].read_bytes()
    with pytest.raises(FileExistsError):
        level("so3", 1, 4, directory)
    assert sources[0].read_bytes() == before


def test_positive_determinants_are_not_closed_under_componentwise_averaging():
    result = admissibility_counterexample()
    assert_allclose(result["source_determinants"], [.3, .3], atol=1e-14)
    assert_allclose(result["lie_determinant"], .3, atol=1e-14)
    assert_allclose(result["componentwise_determinant"], -2.1025, atol=1e-14)


def test_published_receipt_matches_sources_metrics_and_figures():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "docs/convergence-receipt.json").read_text())
    for relative, expected in receipt["provenance"]["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root / "docs/convergence-metrics.csv").read_bytes()).hexdigest() == receipt["evidence_sha256"]["metrics.csv"]
    for name, expected in receipt["figures_sha256"].items():
        assert hashlib.sha256((root / "docs/figures" / name).read_bytes()).hexdigest() == expected
    assert len(receipt["checks"]) == 18
    assert all(check["passed"] for check in receipt["checks"])
