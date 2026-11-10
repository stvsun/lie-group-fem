"""Assemble two tetrahedra and recover SO(3) and GL+(3) integration-point fields."""

import numpy as np
from lie_fem import SO3Chart, GL3Chart, recover, tetrahedron_shape, tetrahedron_quadrature
from _common import VERTICES, rotation_field, deformation_field, save


def main():
    vertices = np.vstack([VERTICES, [1., 1., 1.]])
    connectivity = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    rows, points, measures = [], [], []
    for nodes in connectivity:
        x, w = tetrahedron_quadrature(vertices[nodes])
        local = np.zeros((4, len(vertices)))
        local[:, nodes] = tetrahedron_shape(vertices[nodes], x)
        rows.append(local)
        points.append(x)
        measures.append(w)
    shape, points, measures = np.vstack(rows), np.vstack(points), np.concatenate(measures)
    metrics, arrays = {}, {}
    for name, chart, analytic in [("so3", SO3Chart(), rotation_field), ("gl3", GL3Chart(), deformation_field)]:
        exact_samples = analytic(points)
        exact_fit = recover(exact_samples, shape, measures, chart)
        exact_error = float(np.max(np.abs(exact_fit.nodal_values - analytic(vertices))))
        assert exact_error < 1e-12
        # Add a smooth, unresolved component to expose a nonzero fit residual.
        samples = exact_samples.copy()
        angle = .04 * points[:, 0] * points[:, 1]
        for q, a in enumerate(angle):
            c, s = np.cos(a), np.sin(a)
            samples[q] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ samples[q]
        fitted = recover(samples, shape, measures, chart)
        assert fitted.stationarity_residual < 1e-12
        assert fitted.weighted_rms > 1e-5
        metrics[name] = {
            "affine_field_maximum_error": exact_error,
            "nonlinear_field_weighted_rms": fitted.weighted_rms,
            "stationarity_residual": fitted.stationarity_residual,
            "weighted_design_condition": fitted.design_condition,
        }
        arrays.update({name + "_source": samples, name + "_exact_source": exact_samples,
                       name + "_nodal_coordinates": fitted.coordinates,
                       name + "_nodal_values": fitted.nodal_values,
                       name + "_reconstructed": fitted.evaluate(shape)})
    save("variational-recovery", metrics, vertices=vertices, connectivity=connectivity,
         points=points, measures=measures, shape=shape, **arrays)


if __name__ == "__main__":
    main()
