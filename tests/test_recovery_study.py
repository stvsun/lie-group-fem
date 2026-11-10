import numpy as np
import pytest
from numpy.testing import assert_allclose

from lie_fem import recover, tetrahedron_shape
from studies.interpolation import axis_rotation, chart_for, cube_mesh, manufactured, tetrahedron_rule
from studies.recovery_transfer import (
    consistent_mass, evaluate_coordinates, evaluate_stage, fit, geometry, locate_source, target_mesh,
)


def dense_design(n, order):
    vertices, cells = cube_mesh(n)
    shape, reference_weights = tetrahedron_rule(order)
    points = np.einsum("qa,eai->eqi", shape, vertices[cells]).reshape(-1, 3)
    determinant, _ = geometry(vertices, cells)
    weights = (determinant[:, None]*reference_weights).ravel()
    design = np.zeros((len(points), len(vertices)))
    for index, cell in enumerate(cells):
        design[index*len(shape):(index+1)*len(shape), cell] = shape
    return design, weights, points


@pytest.mark.parametrize("case", ["so3", "gl3_compressible", "gl3_isochoric"])
def test_sparse_recovery_against_public_svd_and_independent_component_fit(case):
    design, weights, points = dense_design(1, 4)
    source = manufactured(points, case)
    result = fit(1, case, 4)
    expected = recover(source, design, weights, chart_for(case))
    assert_allclose(result["lie"], expected.coordinates, atol=3e-14)
    ordinary = np.linalg.lstsq(np.sqrt(weights)[:, None]*design,
                              np.sqrt(weights)[:, None]*source.reshape(-1, 9), rcond=None)[0]
    assert_allclose(result["componentwise"], ordinary, atol=3e-14)
    for method in ("lie", "componentwise"):
        assert result["diagnostics"][method]["relative_normal_residual"] < 1e-13


def test_analytic_mass_equals_independent_quadrature_assembly():
    vertices, cells = cube_mesh(2)
    design, weights, _ = dense_design(2, 4)
    assert_allclose(consistent_mass(vertices, cells).toarray(), design.T@(weights[:, None]*design), atol=2e-16)
    assert np.linalg.eigvalsh(consistent_mass(vertices, cells).toarray()).min() > 0


@pytest.mark.parametrize("n", [1, 2, 4])
def test_locator_reproduces_points_and_matches_independent_tetrahedron_solve(n):
    random = np.random.default_rng(93021)
    points = np.vstack([random.uniform(size=(24, 3)), np.eye(3), np.zeros(3), np.ones(3), [.5, .5, .5]])
    vertices, _ = cube_mesh(n)
    nodes, weights = locate_source(n, points)
    assert_allclose(weights.sum(axis=1), 1., atol=1e-15)
    assert np.all(weights >= 0)
    assert_allclose(np.einsum("qa,qai->qi", weights, vertices[nodes]), points, atol=5e-15)
    for point, cell, weight in zip(points, nodes, weights):
        order = np.arange(4)
        if np.linalg.det((vertices[cell[1:]]-vertices[cell[0]]).T) < 0:
            order[[1, 2]] = order[[2, 1]]
        expected = tetrahedron_shape(vertices[cell[order]], point[None])[0]
        assert_allclose(weight[order], expected, atol=5e-15)
    with pytest.raises(ValueError, match="outside"):
        locate_source(n, [[-.01, .1, .2]])


def test_target_is_nonnested_with_positive_volume_and_matching_boundary():
    vertices, cells = target_mesh(2)
    determinant, h = geometry(vertices, cells)
    assert_allclose(determinant.sum()/6, 1., atol=1e-14)
    assert h > 0 and np.all(determinant > 0)
    assert np.any(np.abs(2*vertices-np.round(2*vertices)) > .01)
    source_vertices, source_cells = cube_mesh(1)
    target_vertices, target_cells = target_mesh(1)
    # For n=1 node coordinates agree, but physical tetrahedra differ.
    source_sets = {tuple(sorted(map(tuple, source_vertices[cell]))) for cell in source_cells}
    target_sets = {tuple(sorted(map(tuple, target_vertices[cell]))) for cell in target_cells}
    assert source_sets != target_sets


@pytest.mark.parametrize("case", ["so3", "gl3_isochoric"])
def test_affine_chart_recovery_and_full_transfer_are_exact(case):
    def field(points, unused_case):
        x, y, z = points.T
        r = axis_rotation(.2+.1*x+.05*y+.1*z, 2)
        if case == "so3":
            return r
        return r*np.exp(np.column_stack([.1*x, .2*y, -.1*x-.2*y]))[:, None, :]
    result = fit(2, case, 4, field=field)
    assert_allclose(result["chart"].decode(result["lie"]), field(result["vertices"], case), atol=2e-14)
    vertices, cells = target_mesh(2)
    nodes, weights = locate_source(2, vertices)
    target = evaluate_coordinates(result["lie"], nodes, weights)
    shape, _ = tetrahedron_rule(4)
    points = np.einsum("qa,eai->eqi", shape, vertices[cells]).reshape(-1, 3)
    coordinates = np.einsum("qa,eak->eqk", shape, target[cells]).reshape(-1, target.shape[1])
    assert_allclose(result["chart"].decode(coordinates), field(points, case), atol=2e-14)


def test_constant_field_is_preserved_by_both_full_transfers():
    matrix = axis_rotation(np.array(.4), 2)@np.diag([1.2, .8, 1.1])
    def constant(points, case):
        return np.broadcast_to(matrix, (len(points), 3, 3)).copy()
    result = fit(2, "gl3_compressible", 4, field=constant)
    vertices, cells = target_mesh(2)
    nodes, weights = locate_source(2, vertices)
    shape, _ = tetrahedron_rule(4)
    for method in ("lie", "componentwise"):
        target = evaluate_coordinates(result[method], nodes, weights)
        coordinates = np.einsum("qa,eak->eqk", shape, target[cells]).reshape(-1, target.shape[1])
        values = result["chart"].decode(coordinates) if method == "lie" else coordinates.reshape(-1, 3, 3)
        assert_allclose(values, np.broadcast_to(matrix, values.shape), atol=3e-14)


@pytest.mark.parametrize("case", ["so3", "gl3_compressible", "gl3_isochoric"])
def test_recovery_and_transfer_refine_and_transfer_increment_is_nonzero(case):
    coarse, fine = fit(1, case, 4), fit(2, case, 4)
    for stage in ("recovery", "transfer"):
        a, b = evaluate_stage(coarse, stage, 5), evaluate_stage(fine, stage, 5)
        for x, y in zip(a, b):
            assert y["relative_l2"] < x["relative_l2"]
            assert_allclose(y["volume"], 1., atol=2e-14)
            if stage == "transfer":
                assert y["sampled_transfer_increment"] > 1e-4
        if case == "so3":
            assert b[0]["orthogonality_rms"] < 1e-13
        if case == "gl3_isochoric":
            assert b[0]["relative_determinant_l2"] < 1e-13


def test_raw_transfer_keeps_intermediate_fields_and_refuses_overwrite(tmp_path):
    result = fit(1, "so3", 4, tmp_path/"source")
    directory = tmp_path/"transfer"
    metrics = evaluate_stage(result, "transfer", 5, directory)
    raw = directory/"evaluation-000000.npz"
    before = raw.read_bytes()
    with np.load(raw, allow_pickle=False) as data:
        error2 = np.sum((data["lie"]-data["exact"])**2, axis=(1, 2))
        norm2 = np.sum(data["exact"]**2, axis=(1, 2))
        error = np.sqrt((data["measures"]@error2)/(data["measures"]@norm2))
        assert_allclose(error, metrics[0]["relative_l2"], atol=1e-16)
        assert "lie_direct" in data and "source_nodes" in data and "source_weights" in data
    with pytest.raises(FileExistsError):
        evaluate_stage(result, "transfer", 5, directory)
    assert raw.read_bytes() == before


def test_published_recovery_receipt_matches_code_metrics_and_figures():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root/"docs/recovery-transfer-receipt.json").read_text())
    for relative, expected in receipt["provenance"]["source_sha256"].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root/"docs/recovery-transfer-metrics.csv").read_bytes()).hexdigest() == receipt["evidence_sha256"]["metrics.csv"]
    for name, expected in receipt["figures_sha256"].items():
        assert hashlib.sha256((root/"docs/figures"/name).read_bytes()).hexdigest() == expected
    assert len(receipt["checks"]) == 132
    assert all(c["passed"] for c in receipt["checks"])
