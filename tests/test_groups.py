import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm, polar

from lie_fem import GL3Chart, SO3Chart, interpolate


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@pytest.mark.parametrize("angle", [0., 1e-12, 0.4, 2.1, np.pi - 1e-5])
def test_so3_against_independent_matrix_exponential(angle):
    axis = np.array([1., -2., 3.]) / np.sqrt(14)
    reference = rz(0.7)
    vector = angle * axis
    truth = reference @ expm(skew(vector))
    chart = SO3Chart(reference)
    assert_allclose(chart.decode(vector[None])[0], truth, atol=2e-14)
    assert_allclose(chart.encode(truth[None])[0], vector, atol=2e-14)


def test_rotation_interpolation_is_analytic_and_preserves_endpoints():
    angles = np.array([-0.5, 0.8, 0.3])
    values = np.array([rz(a) for a in angles])
    shape = np.vstack([np.eye(3), [0.2, 0.3, 0.5]])
    mapped = interpolate(values, shape, SO3Chart())
    assert_allclose(mapped, [rz(a) for a in shape @ angles], atol=2e-14)
    assert_allclose(mapped.transpose(0, 2, 1) @ mapped, np.broadcast_to(np.eye(3), mapped.shape), atol=2e-14)


@pytest.mark.parametrize("chart", [SO3Chart(), GL3Chart()])
def test_noncommuting_fields_are_left_equivariant_with_reference(chart):
    rotations = np.array([expm(skew(v)) for v in [[0.2, 0.1, -0.1], [-0.4, 0.1, 0.3]]])
    values = rotations
    if isinstance(chart, GL3Chart):
        values = rotations @ np.array([np.diag([1.2, 0.8, 1.]), [[1.1, .2, 0], [.2, 1.3, .1], [0, .1, .9]]])
    observer = expm(skew([0.3, -0.2, 0.6]))
    shape = [[0.35, 0.65]]
    assert_allclose(
        interpolate(observer @ values, shape, type(chart)(observer @ chart.reference)),
        observer @ interpolate(values, shape, chart), atol=3e-14,
    )


def test_polar_chart_against_independent_symmetric_matrix_functions():
    logarithm = np.array([[.2, .1, -.04], [.1, -.3, .08], [-.04, .08, .15]])
    rotation = expm(skew([.2, -.4, .1]))
    gradient = rotation @ expm(logarithm)
    chart = GL3Chart()
    coordinates = chart.encode(gradient[None])
    expected = [0.2, -0.4, 0.1, .2, -.3, .15, np.sqrt(2)*.08, np.sqrt(2)*-.04, np.sqrt(2)*.1]
    assert_allclose(coordinates[0], expected, atol=2e-14)
    assert_allclose(chart.decode(coordinates)[0], gradient, atol=2e-14)
    r_oracle, u_oracle = polar(gradient)
    assert_allclose(r_oracle, rotation, atol=2e-14)
    assert_allclose(logm(u_oracle), logarithm, atol=2e-14)
    assert_allclose(np.linalg.norm(coordinates[0, 3:]), np.linalg.norm(logarithm), atol=2e-14)


@pytest.mark.parametrize("stretch", [[1., 1., 1.], [2., 2., 0.7], [1e-4, 1., 10.]])
def test_repeated_and_distinct_singular_values(stretch):
    basis = expm(skew([.3, .4, -.2]))
    gradient = rz(.6) @ basis @ np.diag(stretch) @ basis.T
    chart = GL3Chart()
    assert_allclose(chart.decode(chart.encode(gradient[None]))[0], gradient, atol=2e-13)


def test_gl_positive_determinant_and_geometric_volume_interpolation():
    values = np.array([rz(-1.) @ np.diag([2., 1., .7]), rz(1.) @ np.diag([.8, 1.3, 1.1])])
    shape = np.column_stack([np.linspace(0, 1, 41), np.linspace(1, 0, 41)])
    mapped = interpolate(values, shape, GL3Chart())
    expected = np.exp(shape @ np.log(np.linalg.det(values)))
    assert np.all(np.linalg.det(mapped) > 0)
    assert_allclose(np.linalg.det(mapped), expected, rtol=2e-14)
    assert_allclose(mapped[[0, -1]], values[[1, 0]], atol=2e-14)


def test_common_reference_handles_rotations_straddling_global_pi():
    values = np.array([rz(np.deg2rad(170)), rz(np.deg2rad(190))])
    mapped = interpolate(values, [[.5, .5]], SO3Chart(rz(np.pi)))
    assert_allclose(mapped[0], rz(np.pi), atol=2e-14)


@pytest.mark.parametrize("value", [np.diag([-1., 1., 1.]), np.diag([0., 1., 1.]), np.diag([1e-14, 1., 1.])])
def test_gl_rejects_invalid_gradients(value):
    with pytest.raises(ValueError):
        GL3Chart().encode(value[None])


@pytest.mark.parametrize("value", [np.diag([1., 1., -1.]), 1.01*np.eye(3), np.full((3, 3), np.nan)])
def test_so3_does_not_silently_repair_invalid_rotations(value):
    with pytest.raises(ValueError):
        SO3Chart().encode(value[None])


@pytest.mark.parametrize("angle", [np.pi, np.pi - 1e-8])
def test_branch_cut_is_rejected(angle):
    with pytest.raises(ValueError, match="branch cut"):
        SO3Chart().encode(rz(angle)[None])
    with pytest.raises(ValueError, match="branch cut"):
        SO3Chart().decode([[0, 0, angle]])


def test_extrapolation_must_stay_in_chart():
    with pytest.raises(ValueError, match="branch cut"):
        interpolate(np.array([rz(0), rz(2)]), [[-1, 2]], SO3Chart())


@pytest.mark.parametrize("weights", [[[.2, .2]], [[np.nan, .5]], [1., 0.]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        interpolate(np.array([np.eye(3), np.eye(3)]), weights, SO3Chart())


@pytest.mark.parametrize("coordinates", [[[0]*3 + [800, 800, 800, 0, 0, 0]], [[0]*3 + [0, 0, -40, 0, 0, 0]]])
def test_unrepresentable_stretch_rejected(coordinates):
    with pytest.raises(ValueError):
        GL3Chart().decode(coordinates)


def test_input_arrays_are_not_modified():
    reference = rz(.1)
    before = reference.copy()
    chart = SO3Chart(reference)
    reference[:] = 0
    assert_allclose(chart.reference, before)
    values = np.array([rz(.2), rz(.4)])
    original = values.copy()
    interpolate(values, [[.2, .8]], chart)
    assert_allclose(values, original)
    with pytest.raises(ValueError):
        SO3Chart().encode(np.eye(3, dtype=complex)[None])


@pytest.mark.parametrize("chart,dimension", [(SO3Chart(), 3), (GL3Chart(), 9)])
def test_decode_accepts_readonly_strided_coordinates(chart, dimension):
    storage = np.zeros((2, 2*dimension))
    coordinates = storage[:, ::2]
    coordinates[:, 2] = .2
    coordinates.setflags(write=False)
    original = coordinates.copy()
    assert not coordinates.flags.c_contiguous
    assert_allclose(chart.decode(coordinates), [rz(.2), rz(.2)], atol=2e-14)
    assert_allclose(coordinates, original)
    assert not coordinates.flags.writeable
