#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split regression tests for the three pipeline steps:

1) registration — 4 frames (3 Elastix steps) vs registration_golden.npz
2) preprocess — grid mask/params vs preprocess_golden.npz (64-frame window)
3) analysis — reads tests/expected/analysis_inputs.npz (warped float32 + preprocess
   arrays; no Elastix). That file is gitignored (~tens of MB); create it locally with
   --regen-golden. Small metrics baselines: analysis_golden.npz (committed).

  python -m pytest tests/test_pipeline_regression.py
  python -m pytest tests/test_pipeline_regression.py --regen-golden
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy import io

import calcium_analysis as ca
import imregistration as imreg
import imutils as imu
import plotutils as pu

FRAMERATE = 65.18
VIDEO_START = 100
N_FRAMES_REGISTRATION = 4
VIDEOTHRESH_REG = (VIDEO_START, VIDEO_START + N_FRAMES_REGISTRATION)
N_FRAMES_ANALYSIS = 64
VIDEOTHRESH_ANALYSIS = (VIDEO_START, VIDEO_START + N_FRAMES_ANALYSIS)

TISSUE_DIV_X = 2
TISSUE_DIV_Y = 6
WIDTH_FACTOR = 1 / 10

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = REPO_ROOT / "example" / "data"
EXPECTED = REPO_ROOT / "tests" / "expected"
REGISTRATION_GOLDEN = EXPECTED / "registration_golden.npz"
PREPROCESS_GOLDEN = EXPECTED / "preprocess_golden.npz"
ANALYSIS_GOLDEN = EXPECTED / "analysis_golden.npz"
ANALYSIS_INPUTS = EXPECTED / "analysis_inputs.npz"


def example_mat_path() -> Path:
    pattern = str(EXAMPLE_DATA / "*.mat")
    mats = sorted(
        p
        for p in glob.glob(pattern)
        if "_warped" not in os.path.basename(p)
    )
    if not mats:
        pytest.skip(f"No .mat files found under {EXAMPLE_DATA}")
    return Path(mats[0])


def _output_dir_for_sample(sample: str) -> Path:
    return REPO_ROOT / "tests" / "output" / sample


def run_registration(
    mat_path: Path, out_dir: Path, videothresh: tuple[int, int]
) -> np.ndarray:
    sample = mat_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    data = imu.load_data(str(mat_path), videothresh=videothresh)
    warped_data, _ = imreg.register_all_frames(data)
    io.savemat(str(out_dir / f"{sample}_warped.mat"), {"warped_data": warped_data})
    return warped_data


def run_preprocess_grid(
    mat_path: Path, out_dir: Path, videothresh: tuple[int, int]
) -> None:
    sample = mat_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    data = imu.load_data(str(mat_path), videothresh=videothresh)
    mask = imu.get_tissue_mask(data, interactive=False)
    data_2d, mask_2d = imu.stack_first_frame_for_rotate(data, mask)
    imu.rotate_data(data_2d, mask_2d)
    is_one_region = True
    region_params = [TISSUE_DIV_X, TISSUE_DIV_Y]
    np.savez(
        str(out_dir / f"{sample}_preprocessing.npz"),
        mask=mask.squeeze(),
        region_params=np.array(region_params),
        type=np.array([is_one_region]),
    )


def _noop_savefig(*_args, **_kwargs):
    return None


def run_analyze(sample: str, out_dir: Path) -> dict:
    preprocess_info = np.load(str(out_dir / f"{sample}_preprocessing.npz"))
    is_one_region = bool(preprocess_info["type"][0])
    region_params = preprocess_info["region_params"]
    mask = preprocess_info["mask"]

    warped_data = io.loadmat(str(out_dir / f"{sample}_warped.mat"))["warped_data"]
    warped_data, mask = imu.rotate_data_cv2(warped_data, mask)
    first_frame = warped_data[0]

    if is_one_region:
        regions = imu.divide_tissue_in_regions(
            mask=mask, nx=region_params[0], ny=region_params[1]
        )
    else:
        regions = imu.apply_threshold(
            final_threshold=region_params[0],
            data=first_frame,
            tissue_mask=mask,
        )

    tissue_calcium_trace = None
    if is_one_region:
        tissue_trace = imu.evaluate_regional_intensities(
            warped_data, mask.astype(int)
        )[:, 0]
        filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace(tissue_trace)
        if len(max_peaks_idx) <= 2:
            bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = (
                0, 0, 0, 0, 0,
            )
        else:
            bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = (
                ca.trace_outputs(
                    filtered_trace, max_peaks_idx, min_peaks_idx, FRAMERATE
                )
            )
        tissue_calcium_trace = ca.CalciumTrace(
            filtered_trace,
            max_peaks_idx,
            min_peaks_idx,
            0,
            bpm,
            bpm_std,
            timing_irregularity,
            upstroke_time,
            amplitude,
        )

    traces = imu.evaluate_regional_intensities(warped_data, regions)
    calcium_traces = []
    valid_regions = []
    filtered_traces = []
    max_peaks = []
    min_peaks = []
    for i, trace in enumerate(traces.T):
        filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace_fft(
            trace, framerate=FRAMERATE, width_factor=WIDTH_FACTOR
        )
        filtered_traces.append(filtered_trace)
        max_peaks.append(max_peaks_idx)
        min_peaks.append(min_peaks_idx)
        if len(max_peaks_idx) <= 2:
            continue
        bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = (
            ca.trace_outputs(
                filtered_trace, max_peaks_idx, min_peaks_idx, FRAMERATE
            )
        )
        ctrace = ca.CalciumTrace(
            filtered_trace,
            max_peaks_idx,
            min_peaks_idx,
            i + 1,
            bpm,
            bpm_std,
            timing_irregularity,
            upstroke_time,
            amplitude,
        )
        calcium_traces.append(ctrace)
        valid_regions.append(ctrace.region)

    filtered_traces_arr = np.array(filtered_traces)
    synchronicity = float(np.mean(np.corrcoef(filtered_traces_arr)))

    orig_savefig = plt.savefig
    fig_savefig = plt.Figure.savefig
    try:
        plt.savefig = _noop_savefig
        plt.Figure.savefig = lambda self, *a, **k: None

        fig1, fig2 = pu.plot_regions_traces(
            first_frame, regions, filtered_traces, framerate=FRAMERATE
        )
        fig1.savefig(str(out_dir / f"{sample}_all_regions.png"), dpi=300)
        fig2.savefig(str(out_dir / f"{sample}_all_traces.png"), dpi=300)

        valid_mask = np.isin(regions, valid_regions)
        regions_work = np.copy(regions)
        regions_work[~valid_mask] = 0
        aux_regions = np.copy(regions_work)
        for i, ctrace in enumerate(calcium_traces):
            regions_work[aux_regions == ctrace.region] = i + 1
            ctrace.region = i + 1

        clean_traces = np.array([t.trace for t in calcium_traces])
        fig1, fig2 = pu.plot_regions_traces(
            first_frame, regions_work, clean_traces, framerate=FRAMERATE
        )
        fig1.savefig(str(out_dir / f"{sample}_regular_regions.png"), dpi=300)
        fig2.savefig(str(out_dir / f"{sample}_regular_traces.png"), dpi=300)
        plt.close("all")
    finally:
        plt.savefig = orig_savefig
        plt.Figure.savefig = fig_savefig

    region_metrics = np.array(
        [
            [
                c.bpm,
                c.bpm_std,
                c.timing_irregularity,
                c.upstroke_time,
                c.amplitude,
            ]
            for c in calcium_traces
        ],
        dtype=np.float64,
    )

    tissue_vec = np.array(
        [
            tissue_calcium_trace.bpm,
            tissue_calcium_trace.bpm_std,
            tissue_calcium_trace.timing_irregularity,
            tissue_calcium_trace.upstroke_time,
            tissue_calcium_trace.amplitude,
        ],
        dtype=np.float64,
    )

    return {
        "warped_data": warped_data,
        "mask": np.asarray(mask),
        "region_params": np.asarray(region_params, dtype=np.float64),
        "is_one_region": np.array([is_one_region]),
        "synchronicity": np.array([synchronicity]),
        "tissue_metrics": tissue_vec,
        "region_metrics": region_metrics,
        "n_valid_regions": np.array([len(calcium_traces)]),
    }


def _registration_fingerprint(warped: np.ndarray) -> np.ndarray:
    w = warped.astype(np.float64)
    t, h, wd = w.shape
    st = max(1, t // 8)
    sh = max(1, h // 48)
    sw = max(1, wd // 48)
    return w[::st, ::sh, ::sw]


def _snapshot_registration(warped: np.ndarray) -> dict:
    wf = warped.astype(np.float64)
    return {
        "reg_warped_shape": np.array(wf.shape),
        "reg_warped_mean": np.array([wf.mean()]),
        "reg_warped_std": np.array([wf.std()]),
        "reg_warped_fingerprint": _registration_fingerprint(warped),
    }


def _assert_registration_close(actual: dict, expected: dict) -> None:
    np.testing.assert_array_equal(
        actual["reg_warped_shape"], expected["reg_warped_shape"]
    )
    np.testing.assert_allclose(
        actual["reg_warped_mean"],
        expected["reg_warped_mean"],
        rtol=1e-5,
        atol=0.05,
    )
    np.testing.assert_allclose(
        actual["reg_warped_std"],
        expected["reg_warped_std"],
        rtol=1e-4,
        atol=0.05,
    )
    np.testing.assert_allclose(
        actual["reg_warped_fingerprint"],
        expected["reg_warped_fingerprint"],
        rtol=1e-3,
        atol=0.5,
        err_msg="registration fingerprint mismatch (Elastix / platform variance)",
    )


def _assert_preprocess_close(actual: dict, expected: dict) -> None:
    np.testing.assert_array_equal(
        actual["preprocess_is_one_region"], expected["preprocess_is_one_region"]
    )
    np.testing.assert_allclose(
        actual["preprocess_region_params"],
        expected["preprocess_region_params"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        actual["preprocess_mask"],
        expected["preprocess_mask"],
        rtol=0,
        atol=0,
    )


def _assert_analysis_close(actual: dict, expected: dict) -> None:
    np.testing.assert_array_equal(
        actual["rotated_warped_shape"], expected["rotated_warped_shape"]
    )
    np.testing.assert_allclose(
        actual["synchronicity"],
        expected["synchronicity"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual["tissue_metrics"],
        expected["tissue_metrics"],
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_equal(actual["n_valid_regions"], expected["n_valid_regions"])
    np.testing.assert_allclose(
        actual["region_metrics"],
        expected["region_metrics"],
        rtol=1e-5,
        atol=1e-4,
    )


def test_regenerate_all_golden_files():
    mat_path = example_mat_path()
    sample = mat_path.stem
    imreg.NCORES = 2
    EXPECTED.mkdir(parents=True, exist_ok=True)

    data_reg = imu.load_data(str(mat_path), videothresh=VIDEOTHRESH_REG)
    warped_short, _ = imreg.register_all_frames(data_reg)
    np.savez_compressed(
        str(REGISTRATION_GOLDEN),
        **_snapshot_registration(warped_short),
    )

    tmp_pre = REPO_ROOT / "tests" / "output" / "_golden_preprocess"
    tmp_pre.mkdir(parents=True, exist_ok=True)
    run_preprocess_grid(mat_path, tmp_pre, videothresh=VIDEOTHRESH_ANALYSIS)
    pz_pre = np.load(str(tmp_pre / f"{sample}_preprocessing.npz"))
    np.savez_compressed(
        str(PREPROCESS_GOLDEN),
        preprocess_mask=np.asarray(pz_pre["mask"]),
        preprocess_region_params=np.asarray(pz_pre["region_params"]),
        preprocess_is_one_region=pz_pre["type"],
    )

    tmp_full = REPO_ROOT / "tests" / "output" / "_golden_full"
    tmp_full.mkdir(parents=True, exist_ok=True)
    data_full = imu.load_data(str(mat_path), videothresh=VIDEOTHRESH_ANALYSIS)
    warped_full, _ = imreg.register_all_frames(data_full)
    io.savemat(str(tmp_full / f"{sample}_warped.mat"), {"warped_data": warped_full})
    run_preprocess_grid(mat_path, tmp_full, videothresh=VIDEOTHRESH_ANALYSIS)
    pz_full = np.load(str(tmp_full / f"{sample}_preprocessing.npz"))
    np.savez_compressed(
        str(ANALYSIS_INPUTS),
        warped_data=warped_full.astype(np.float32),
        mask=pz_full["mask"],
        region_params=pz_full["region_params"],
        preprocessing_type=pz_full["type"],
    )

    analyzed = run_analyze(sample, tmp_full)
    np.savez_compressed(
        str(ANALYSIS_GOLDEN),
        rotated_warped_shape=np.array(analyzed["warped_data"].shape),
        synchronicity=analyzed["synchronicity"],
        tissue_metrics=analyzed["tissue_metrics"],
        region_metrics=analyzed["region_metrics"],
        n_valid_regions=analyzed["n_valid_regions"],
    )


def test_registration_matches_golden():
    mat_path = example_mat_path()
    imreg.NCORES = 2
    data = imu.load_data(str(mat_path), videothresh=VIDEOTHRESH_REG)
    warped, _ = imreg.register_all_frames(data)
    actual = _snapshot_registration(warped)
    if not REGISTRATION_GOLDEN.is_file():
        pytest.fail(
            f"Missing {REGISTRATION_GOLDEN}. Run:\n"
            f"  python -m pytest tests/test_pipeline_regression.py --regen-golden"
        )
    expected = dict(np.load(str(REGISTRATION_GOLDEN)))
    _assert_registration_close(actual, expected)


def test_preprocess_matches_golden():
    mat_path = example_mat_path()
    sample = mat_path.stem
    out_dir = _output_dir_for_sample(f"{sample}_preprocess_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        child.unlink()
    run_preprocess_grid(mat_path, out_dir, videothresh=VIDEOTHRESH_ANALYSIS)
    pz = np.load(str(out_dir / f"{sample}_preprocessing.npz"))
    actual = {
        "preprocess_mask": np.asarray(pz["mask"]),
        "preprocess_region_params": np.asarray(pz["region_params"]),
        "preprocess_is_one_region": pz["type"],
    }
    if not PREPROCESS_GOLDEN.is_file():
        pytest.fail(
            f"Missing {PREPROCESS_GOLDEN}. Run:\n"
            f"  python -m pytest tests/test_pipeline_regression.py --regen-golden"
        )
    expected = dict(np.load(str(PREPROCESS_GOLDEN)))
    _assert_preprocess_close(actual, expected)


def test_analysis_matches_golden():
    if not ANALYSIS_INPUTS.is_file():
        pytest.fail(
            f"Missing {ANALYSIS_INPUTS}. Run:\n"
            f"  python -m pytest tests/test_pipeline_regression.py --regen-golden"
        )
    mat_path = example_mat_path()
    sample = mat_path.stem
    inputs = dict(np.load(str(ANALYSIS_INPUTS)))

    out_dir = _output_dir_for_sample(f"{sample}_analysis_from_npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        child.unlink()

    warped = np.asarray(inputs["warped_data"], dtype=np.float64)
    io.savemat(str(out_dir / f"{sample}_warped.mat"), {"warped_data": warped})
    np.savez(
        str(out_dir / f"{sample}_preprocessing.npz"),
        mask=inputs["mask"],
        region_params=inputs["region_params"],
        type=inputs["preprocessing_type"],
    )

    analyzed = run_analyze(sample, out_dir)
    actual = {
        "rotated_warped_shape": np.array(analyzed["warped_data"].shape),
        "synchronicity": analyzed["synchronicity"],
        "tissue_metrics": analyzed["tissue_metrics"],
        "region_metrics": analyzed["region_metrics"],
        "n_valid_regions": analyzed["n_valid_regions"],
    }
    if not ANALYSIS_GOLDEN.is_file():
        pytest.fail(
            f"Missing {ANALYSIS_GOLDEN}. Run:\n"
            f"  python -m pytest tests/test_pipeline_regression.py --regen-golden"
        )
    expected = dict(np.load(str(ANALYSIS_GOLDEN)))
    _assert_analysis_close(actual, expected)

