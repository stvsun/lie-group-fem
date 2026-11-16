import numpy as np
from numpy.testing import assert_allclose

from studies.paper_hex import BoxMesh, ring_fields, ring_decode, recovered_ring
from studies.paper_examples import table_examples, scalar_example


def test_q1_box_moments_and_affine_partition():
    mesh = BoxMesh([3, 2, 1], [-1, -2, -3], [1, 2, 3])
    p, w, n = mesh.quadrature(2)
    assert_allclose(w.sum(), 48)
    assert_allclose(n.sum(axis=1), 1)
    assert_allclose(n @ mesh.nodes, p, atol=1e-15)
    assert_allclose(w @ p ** 2, [16, 64, 144], atol=1e-13)


def test_scalar_projection_analytic_global_and_local(tmp_path):
    row = scalar_example(tmp_path, {"counts": [2, 2, 2], "lower": [-1]*3, "upper": [1]*3, "quadrature_order": 2})
    assert row["global_nodal_oracle_max"] < 1e-14
    assert row["local_sample_max"] < 1e-14
    assert_allclose(row["global_L2"], np.sqrt(.5), atol=1e-14)


def test_published_tables_preserve_known_discrepancy(tmp_path):
    rows = table_examples(tmp_path, .00501)
    mismatches = [(r["group"], r["method"]) for r in rows if not r["matches_printed_rounding"]]
    assert mismatches == [("SO", "componentwise")]


def test_ring_recovery_exactness_and_collocation_caveat():
    # One Y cell: two Gauss samples of log stretch fit exactly but off-rule values do not.
    mesh = BoxMesh([4, 1, 1], [-.5, -1/32, -1/32], [.5, 1/32, 1/32])
    p, w, n = mesh.quadrature(2)
    audit, aw, an = mesh.quadrature(5)
    mixed, residual, decoder = recovered_ring(mesh, "mixed")
    assert_allclose(ring_decode(an @ mixed, decoder)[0], ring_fields(audit)[0], atol=1e-14)
    lie, residual, decoder = recovered_ring(mesh, "lie")
    assert_allclose(ring_decode(n @ lie, decoder)[0], ring_fields(p)[0], atol=1e-14)
    assert np.linalg.norm(ring_decode(an @ lie, decoder)[0] - ring_fields(audit)[0]) > .01


def test_polar_direct_and_direct_f_separable_ring_agree():
    mesh = BoxMesh([8, 2, 1], [-.5, -1/32, -1/32], [.5, 1/32, 1/32])
    a, _, _ = recovered_ring(mesh, "direct")
    b, _, _ = recovered_ring(mesh, "direct_F")
    assert_allclose(ring_decode(a, "direct")[0], ring_decode(b, "direct_F")[0], atol=1e-14)


def test_full_paper_run_serializes_verifies_and_refuses_overwrite(tmp_path):
    import json
    import pytest
    pytest.importorskip("matplotlib", reason="full plot replay requires the study extra")
    from studies.paper_examples import run
    output = tmp_path / "new-run"
    report = run(output)
    assert all(report["checks"].values())
    assert not report["exact_published_replication"]
    assert (output / "ring-recovery.npz").exists()
    assert json.loads((output / "manifest.json").read_text())["paper-ring-recovery.png"]
    with pytest.raises(FileExistsError):
        run(output)


def test_published_paper_receipt_matches_code_and_assets():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "docs/paper-receipt.json").read_text())
    for name, expected in (receipt["source_hashes"] | receipt["published_hashes"]).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    report = json.loads((root / "docs/paper-results.json").read_text())
    assert len(report["checks"]) == 14 and all(report["checks"].values())
    assert report["exact_published_replication"] is False
    assert sum(r["matches_published"] for r in report["ring_interpolation"]) == 2
    assert not all(all(r["matches_published"]) for r in report["ring_recovery"])
