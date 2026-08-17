from pathlib import Path

import numpy as np
from scipy import io as scipy_io

from calcium_pipeline import discover_samples, is_primary_input_candidate, load_stack_for_viewer


def test_discover_samples_groups_known_artifacts(tmp_path: Path):
    (tmp_path / "alpha.mat").write_bytes(b"mat-placeholder")
    (tmp_path / "alpha_warped.mat").write_bytes(b"warped-placeholder")
    (tmp_path / "alpha_tissue_mask.tif").write_bytes(b"mask-placeholder")
    (tmp_path / "alpha_output.csv").write_text("a,b\n1,2\n")
    (tmp_path / "alpha_all_regions.png").write_bytes(b"png-placeholder")
    (tmp_path / "beta.tif").write_bytes(b"tif-placeholder")

    samples = discover_samples(tmp_path)

    assert [sample.name for sample in samples] == ["alpha", "beta"]
    alpha = samples[0]
    assert alpha.input_path.name == "alpha.mat"
    assert [artifact.kind for artifact in alpha.artifacts] == [
        "input",
        "warped_stack",
        "mask",
        "summary_csv",
        "regions_png",
    ]


def test_primary_input_candidate_filters_generated_outputs():
    assert is_primary_input_candidate("example.mat")
    assert is_primary_input_candidate("example.nd2")
    assert not is_primary_input_candidate("example_warped.mat")
    assert not is_primary_input_candidate("example_tissue_mask.tif")
    assert not is_primary_input_candidate("example_output.csv")


def test_load_stack_for_viewer_reads_warped_mat(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    mat_path = tmp_path / "demo_warped.mat"
    scipy_io.savemat(mat_path, {"warped_data": data})

    loaded = load_stack_for_viewer(mat_path)

    assert loaded.shape == (3, 4, 2)
    assert np.allclose(loaded, data)
