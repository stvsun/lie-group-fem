import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("example", ["so3_tetrahedron", "gl3_remap", "variational_recovery"])
def test_example_preserves_fields_and_refuses_overwrite(example, tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / example
    command = [sys.executable, str(root / "examples" / (example + ".py")), "--output", str(output)]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    report = json.loads((output / "report.json").read_text())
    assert report["metrics"]
    with np.load(output / "fields.npz", allow_pickle=False) as arrays:
        assert "points" in arrays and "measures" in arrays
        assert all(np.all(np.isfinite(arrays[key])) for key in arrays.files)
    before = (output / "fields.npz").read_bytes()
    repeated = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
    assert (output / "fields.npz").read_bytes() == before
