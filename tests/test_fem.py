import numpy as np
import pytest
from numpy.testing import assert_allclose

from lie_fem import GL3Chart, SO3Chart, recover, tetrahedron_shape, tetrahedron_quadrature

VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def field(points, group):
    angle = .1 + points @ [.2, -.1, .3]
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.zeros((len(points), 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 2, 2] = -s, s, 1
    if group == "so3":
        return rotation
    log_stretch = points @ np.array([[.2, -.1, .1], [.1, .3, -.1], [-.1, .2, .1]])
    return rotation * np.exp(log_stretch)[:, None, :]


def test_tetrahedron_rule_against_exact_moments_and_mass_matrix():
    points, measures = tetrahedron_quadrature(VERTICES)
    shape = tetrahedron_shape(VERTICES, points)
    assert_allclose(sum(measures), 1/6, atol=1e-15)
    assert_allclose(measures @ points, np.full(3, 1/24), atol=1e-15)
    assert_allclose(points.T @ (measures[:, None] * points), (np.ones((3, 3)) + np.eye(3))/120, atol=1e-15)
    assert_allclose(shape.T @ (measures[:, None] * shape), (np.ones((4, 4)) + np.eye(4))/120, atol=1e-15)
    assert_allclose(tetrahedron_shape(VERTICES, VERTICES), np.eye(4), atol=1e-15)


@pytest.mark.parametrize("group,chart", [("so3", SO3Chart()), ("gl3", GL3Chart())])
def test_affine_chart_field_recovery_and_transfer(group, chart):
    points, measures = tetrahedron_quadrature(VERTICES)
    shape = tetrahedron_shape(VERTICES, points)
    result = recover(field(points, group), shape, measures, chart)
    assert_allclose(result.nodal_values, field(VERTICES, group), atol=2e-14)
    targets = np.array([[.25, .25, .25], [.1, .3, .2]])
    assert_allclose(result.evaluate(tetrahedron_shape(VERTICES, targets)), field(targets, group), atol=2e-14)
    assert result.weighted_rms < 2e-14
    assert result.stationarity_residual < 2e-14


def test_overdetermined_variational_stationarity_and_minimum():
    rng = np.random.default_rng(284)
    shape = rng.dirichlet(np.ones(4), size=30)
    measures = rng.uniform(.1, 2., 30)
    angles = .1 + shape @ np.array([.1, -.2, .3, .4]) + .03*rng.normal(size=30)
    values = np.array([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]] for a in angles])
    result = recover(values, shape, measures, SO3Chart())
    # Independent consistent mass equation oracle, not the production SVD solve.
    mass = shape.T @ (measures[:, None] * shape)
    rhs = shape.T @ (measures * angles)
    truth = np.linalg.solve(mass, rhs)
    assert_allclose(result.coordinates[:, 2], truth, atol=2e-14)
    error = shape @ truth - angles
    assert_allclose(result.weighted_rms**2, np.sum(measures * error**2)/sum(measures), atol=1e-16)
    assert result.weighted_rms > 1e-3
    assert result.stationarity_residual < 2e-14
    for _ in range(10):
        displaced = truth + rng.normal(size=4)*.01
        assert np.sum(measures * (shape @ displaced - angles)**2) > np.sum(measures * error**2)


def test_multielement_assembled_recovery():
    vertices = np.vstack([VERTICES, [1., 1., 1.]])
    connectivity = [[0, 1, 2, 3], [1, 2, 3, 4]]
    rows, points, measures = [], [], []
    for nodes in connectivity:
        x, w = tetrahedron_quadrature(vertices[nodes])
        n = np.zeros((4, 5))
        n[:, nodes] = tetrahedron_shape(vertices[nodes], x)
        rows.append(n)
        points.append(x)
        measures.append(w)
    result = recover(field(np.vstack(points), "gl3"), np.vstack(rows), np.concatenate(measures), GL3Chart())
    assert_allclose(result.nodal_values, field(vertices, "gl3"), atol=2e-14)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_projection_independent_of_uniform_measure_units(scale):
    points, measures = tetrahedron_quadrature(VERTICES)
    shape = tetrahedron_shape(VERTICES, points)
    result = recover(field(points, "so3"), shape, scale*measures, SO3Chart())
    assert_allclose(result.nodal_values, field(VERTICES, "so3"), atol=2e-14)


def test_rank_deficient_recovery_is_not_silently_regularized():
    with pytest.raises(ValueError, match="rank-deficient"):
        recover(np.eye(3)[None], [[.25]*4], [1.], SO3Chart())


@pytest.mark.parametrize("weights", [[0]*4, [-1]*4, [np.nan]*4, [1]*3])
def test_invalid_measures(weights):
    points, _ = tetrahedron_quadrature(VERTICES)
    with pytest.raises(ValueError):
        recover(field(points, "so3"), np.eye(4), weights, SO3Chart())


def test_recovered_nodes_outside_chart_rejected():
    # Samples are inside the chart, but the fitted endpoint angle is 4 radians.
    shape = np.array([[.5, .5], [.6, .4]])
    angles = shape @ [0., 4.]
    values = SO3Chart().decode(np.column_stack([np.zeros((2, 2)), angles]))
    with pytest.raises(ValueError, match="branch cut"):
        recover(values, shape, [1, 1], SO3Chart())


def test_invalid_tetrahedra_and_outside_points():
    for vertices in [VERTICES[[0, 2, 1, 3]], np.zeros((4, 3))]:
        with pytest.raises(ValueError):
            tetrahedron_quadrature(vertices)
    with pytest.raises(ValueError, match="outside"):
        tetrahedron_shape(VERTICES, [[1, 1, 1]])


def test_physical_tetrahedron_affine_geometry():
    transform = np.array([[2., .2, 0], [0, 1.3, .1], [0, 0, .7]])
    vertices = VERTICES @ transform.T + [2, -3, 1]
    points, measures = tetrahedron_quadrature(vertices)
    shape = tetrahedron_shape(vertices, points)
    assert_allclose(shape @ vertices, points, atol=2e-14)
    assert_allclose(sum(measures), np.linalg.det(transform)/6, atol=2e-14)
