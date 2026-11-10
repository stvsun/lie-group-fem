"""Interpolate nodal orientations at integration points of a P1 tetrahedron."""

import numpy as np
from lie_fem import SO3Chart, interpolate, tetrahedron_shape, tetrahedron_quadrature
from _common import VERTICES, rotation_field, save


def main():
    points, measures = tetrahedron_quadrature(VERTICES)
    shape = tetrahedron_shape(VERTICES, points)
    source = rotation_field(VERTICES)
    mapped = interpolate(source, shape, SO3Chart())
    truth = rotation_field(points)
    linear = np.einsum("qa,aij->qij", shape, source)
    metrics = {
        "maximum_matrix_error": float(np.max(np.abs(mapped - truth))),
        "maximum_orthogonality_error": float(np.max(np.abs(mapped.transpose(0, 2, 1) @ mapped - np.eye(3)))),
        "componentwise_interpolation_orthogonality_error": float(np.max(np.abs(linear.transpose(0, 2, 1) @ linear - np.eye(3)))),
    }
    assert metrics["maximum_matrix_error"] < 1e-12
    assert metrics["maximum_orthogonality_error"] < 1e-12
    save("so3-tetrahedron", metrics, vertices=VERTICES, points=points, measures=measures,
         shape=shape, source=source, mapped=mapped, analytic=truth, componentwise=linear)


if __name__ == "__main__":
    main()
