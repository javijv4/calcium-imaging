#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run registration -> preprocess (grid) -> analysis for one .mat under example/data.

Usage (from repo root, with PYTHONPATH=src or editable install):
  python example/run_single_pipeline.py
  python example/run_single_pipeline.py /path/to/file.mat
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy import io

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import imregistration as imreg
import imutils as imu

VIDEO_START = 100
N_FRAMES_REG = 4
N_FRAMES_FULL = 64
TISSUE_DIV_X = 2
TISSUE_DIV_Y = 6


def _default_mat() -> Path:
    pattern = str(REPO_ROOT / "example" / "data" / "*.mat")
    mats = sorted(
        p for p in glob.glob(pattern) if "_warped" not in os.path.basename(p)
    )
    if not mats:
        raise FileNotFoundError(f"No .mat files under {REPO_ROOT / 'example' / 'data'}")
    return Path(mats[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file register / preprocess / save outputs.")
    parser.add_argument(
        "mat_path",
        nargs="?",
        default=None,
        help="Input .mat (default: first non-warped under example/data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "example" / "output" / "single_run",
        help="Output directory",
    )
    args = parser.parse_args()
    mat_path = Path(args.mat_path) if args.mat_path else _default_mat()
    if not mat_path.is_file():
        raise SystemExit(f"Not found: {mat_path}")

    sample = mat_path.stem
    out_dir = args.out / sample
    out_dir.mkdir(parents=True, exist_ok=True)

    imreg.NCORES = 2

    # 1) Registration (short window for speed; use full window for production)
    videothresh_reg = (VIDEO_START, VIDEO_START + N_FRAMES_REG)
    data_reg = imu.load_data(str(mat_path), videothresh=videothresh_reg)
    warped_short, _ = imreg.register_all_frames(data_reg)
    print(f"Registered short stack shape (T,H,W): {warped_short.shape}")

    # Full registration for saved warped + analysis
    videothresh_full = (VIDEO_START, VIDEO_START + N_FRAMES_FULL)
    data_full = imu.load_data(str(mat_path), videothresh=videothresh_full)
    warped_full, _ = imreg.register_all_frames(data_full)
    io.savemat(str(out_dir / f"{sample}_warped.mat"), {"warped_data": warped_full})

    # 2) Preprocess (grid): (T, H, W) throughout
    data = imu.load_data(str(mat_path), videothresh=videothresh_full)
    mask = imu.get_tissue_mask(data, interactive=False)
    data_2d, mask_2d = imu.stack_first_frame_for_rotate(data, mask)
    imu.rotate_data(data_2d, mask_2d)
    np.savez(
        str(out_dir / f"{sample}_preprocessing.npz"),
        mask=mask.squeeze(),
        region_params=np.array([TISSUE_DIV_X, TISSUE_DIV_Y]),
        type=np.array([True]),
    )
    print(f"Wrote preprocessing to {out_dir / (sample + '_preprocessing.npz')}")

    # 3) Quick analysis smoke test
    preprocess_info = np.load(str(out_dir / f"{sample}_preprocessing.npz"))
    mask = preprocess_info["mask"]
    region_params = preprocess_info["region_params"]
    warped_data = io.loadmat(str(out_dir / f"{sample}_warped.mat"))["warped_data"]
    warped_data, mask = imu.rotate_data_cv2(warped_data, mask)
    regions = imu.divide_tissue_in_regions(
        mask=mask, nx=region_params[0], ny=region_params[1]
    )
    traces = imu.evaluate_regional_intensities(warped_data, regions)
    print(f"Rotated warped shape: {warped_data.shape}, regions shape: {regions.shape}")
    print(f"Trace matrix shape (frames x regions): {traces.shape}")
    print("Mean cross-correlation of regions:", float(np.mean(np.corrcoef(traces.T))))
    print("Done.")


if __name__ == "__main__":
    main()
