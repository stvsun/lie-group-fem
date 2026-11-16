"""Independent group constraints and analytic special cases, not round trips alone."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from lie_fem import MatrixChart, PositiveRealChart, interpolate, recover


def coordinates(chart, matrix):
    return np.einsum("kij,ij->k", chart.basis, matrix)[None]


@pytest.mark.parametrize("family,n,dimension", [
    ("GL", 3, 9), ("GL+", 3, 9), ("SL", 3, 8), ("O", 3, 3),
    ("SO", 4, 6), ("Sp", 2, 10), ("SE", 3, 6),
])
def test_bases_and_group_constraints(family, n, dimension):
    chart = MatrixChart(family, n)
    assert chart.dimension == dimension
    assert_allclose(np.einsum("kij,lij->kl", chart.basis, chart.basis), np.eye(dimension), atol=1e-15)
    c = np.linspace(-.2, .3, dimension)[None]
    a = chart.decode(c)[0]
    assert_allclose(chart.encode(a[None]), c, atol=2e-14)
    if family in ("SL", "SO", "Sp", "SE"):
        assert_allclose(np.linalg.det(a), 1, atol=2e-14)
    if family in ("O", "SO", "SE"):
        q = a[:n, :n]
        assert_allclose(q.T @ q, np.eye(n), atol=2e-14)
    if family == "Sp":
        j = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
        assert_allclose(a.T @ j @ a, j, atol=2e-14)
    if family == "SE":
        assert_allclose(a[-1], [0, 0, 0, 1], atol=2e-14)


def test_analytic_nilpotent_and_diagonal_exponentials():
    for family in ("GL", "GL+", "SL"):
        chart = MatrixChart(family, 3)
        b = np.zeros((3, 3)); b[0, 1] = .7
        assert_allclose(chart.decode(coordinates(chart, b))[0], np.eye(3) + b, atol=1e-15)
    chart = MatrixChart("SL", 3)
    d = np.array([.2, -.3, .1])
    assert_allclose(chart.decode(coordinates(chart, np.diag(d)))[0], np.diag(np.exp(d)), atol=1e-15)


def test_so3_against_rodrigues_oracle():
    chart = MatrixChart("SO", 3)
    b = np.array([[0, -.5, .3], [.5, 0, -.2], [-.3, .2, 0]])
    expected = Rotation.from_rotvec([.2, .3, .5]).as_matrix()
    assert_allclose(chart.decode(coordinates(chart, b))[0], expected, atol=1e-15)


def test_se_exponential_couples_rotation_translation():
    chart = MatrixChart("SE", 2)
    theta = .7
    b = np.array([[0, -theta, 1.], [theta, 0, 0], [0, 0, 0]])
    expected = np.array([[np.cos(theta), -np.sin(theta), np.sin(theta) / theta],
                         [np.sin(theta), np.cos(theta), (1 - np.cos(theta)) / theta], [0, 0, 1]])
    assert_allclose(chart.decode(coordinates(chart, b))[0], expected, atol=1e-15)


def test_symplectic_shear_oracle():
    chart = MatrixChart("Sp", 2)
    b = np.zeros((4, 4)); b[:2, 2:] = [[.3, .2], [.2, -.1]]
    assert_allclose(chart.decode(coordinates(chart, b))[0], np.eye(4) + b, atol=1e-15)


@pytest.mark.parametrize("family", ["GL", "O"])
def test_negative_component_requires_reference(family):
    reflection = np.diag([-1., 1, 1])
    with pytest.raises(ValueError, match="branch cut"):
        MatrixChart(family, 3).encode(reflection[None])
    chart = MatrixChart(family, 3, reference=reflection)
    assert_allclose(chart.decode(np.zeros((1, chart.dimension)))[0], reflection)
    assert_allclose(chart.encode(reflection[None]), 0)
    with pytest.raises(ValueError):
        chart.encode(np.eye(3)[None])


@pytest.mark.parametrize("family,n", [("GL+", 3), ("SL", 3), ("SO", 3), ("O", 3), ("Sp", 2), ("SE", 3)])
def test_all_families_work_with_recovery(family, n):
    chart = MatrixChart(family, n)
    x = np.array([.1, .3, .6, .9])
    basis = np.column_stack([1 - x, x])
    endpoints = np.vstack([np.zeros(chart.dimension), np.linspace(-.1, .2, chart.dimension)])
    values = chart.decode(basis @ endpoints)
    recovered = recover(values, basis, np.ones(4), chart)
    assert_allclose(recovered.coordinates, endpoints, atol=3e-14)
    target = np.array([[.6, .4], [-.2, 1.2]])
    assert_allclose(recovered.evaluate(target), chart.decode(target @ endpoints), atol=3e-14)


def test_positive_scalar_exact_geometric_mean_and_recovery():
    chart = PositiveRealChart()
    assert_allclose(interpolate([[.9], [.1]], [[.5, .5], [-.5, 1.5]], chart), [[.3], [1/30]], atol=1e-15)
    n = np.array([[.8, .2], [.2, .8], [.5, .5]])
    c = np.array([[.1], [.8]])
    assert_allclose(recover(np.exp(n @ c), n, [1, 1, 1], chart).coordinates, c, atol=1e-15)
    for bad in ([[0]], [[-1]], [[np.nan]], [[1j]]):
        with pytest.raises(ValueError):
            chart.encode(bad)
    with pytest.raises(ValueError):
        chart.decode([[1000]])


def test_no_silent_constraint_repair_or_branch_wrapping():
    with pytest.raises(ValueError):
        MatrixChart("SO", 3).encode((np.eye(3) * 1.01)[None])
    with pytest.raises(ValueError):
        MatrixChart("SL", 3).encode((np.eye(3) * 1.01)[None])
    with pytest.raises(ValueError):
        MatrixChart("Sp", 2).encode(np.diag([1, 1, 1, 1.1])[None])
    chart = MatrixChart("SO", 2)
    with pytest.raises(ValueError, match="branch"):
        chart.decode(coordinates(chart, np.array([[0, -3.2], [3.2, 0]])))
    with pytest.raises(ValueError, match="branch"):
        chart.encode((-np.eye(2))[None])
    with pytest.raises(ValueError):
        MatrixChart("GL", 2).encode(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        MatrixChart("GL+", 2).decode([[1000, 0, 0, 1000]])


def test_encode_and_decode_share_the_same_branch_margin():
    theta = np.pi - 5e-8
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    with pytest.raises(ValueError, match="branch"):
        MatrixChart("SO", 2).encode(r[None])


def test_classical_example_preserves_raw_data_and_refuses_overwrite(tmp_path):
    from examples.classical_groups import run
    import json
    output = tmp_path / "groups"
    run(output)
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] and len(report["examples"]) == 8
    before = (output / "fields.npz").read_bytes()
    with np.load(output / "fields.npz", allow_pickle=False) as raw:
        assert len([key for key in raw.files if key.endswith("_target")]) == 8
        assert all(np.all(np.isfinite(raw[key])) for key in raw.files)
    with pytest.raises(FileExistsError):
        run(output)
    assert (output / "fields.npz").read_bytes() == before
