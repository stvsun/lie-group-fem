"""Map an admissible tensor history field to four refined tetrahedra.

This is a field transfer demonstration, not a displacement/compatibility solve.
The manufactured unimodular tensors can represent a plastic internal map.
"""

import numpy as np
from lie_fem import GL3Chart, interpolate, tetrahedron_shape, tetrahedron_quadrature
from _common import VERTICES, deformation_field, save


def main():
    refined = np.vstack([VERTICES, VERTICES.mean(axis=0)])
    connectivity = []
    points, measures = [], []
    for opposite in range(4):
        nodes = [i for i in range(4) if i != opposite] + [4]
        vertices = refined[nodes]
        if np.linalg.det((vertices[1:] - vertices[0]).T) < 0:
            nodes[0], nodes[1] = nodes[1], nodes[0]
        x, w = tetrahedron_quadrature(refined[nodes])
        connectivity.append(nodes)
        points.append(x)
        measures.append(w)
    points, measures = np.vstack(points), np.concatenate(measures)
    shape = tetrahedron_shape(VERTICES, points)
    source = deformation_field(VERTICES)
    mapped = interpolate(source, shape, GL3Chart())
    truth = deformation_field(points)
    metrics = {
        "target_elements": 4,
        "maximum_matrix_error": float(np.max(np.abs(mapped - truth))),
        "minimum_determinant": float(np.min(np.linalg.det(mapped))),
        "maximum_isochoric_error": float(np.max(np.abs(np.linalg.det(mapped) - 1))),
        "target_volume": float(np.sum(measures)),
    }
    assert metrics["maximum_matrix_error"] < 1e-12
    assert metrics["maximum_isochoric_error"] < 1e-12
    assert abs(metrics["target_volume"] - 1/6) < 1e-14
    save("gl3-remap", metrics, source_vertices=VERTICES, target_vertices=refined,
         target_connectivity=connectivity, points=points, measures=measures,
         shape=shape, source=source, mapped=mapped, analytic=truth)


if __name__ == "__main__":
    main()
