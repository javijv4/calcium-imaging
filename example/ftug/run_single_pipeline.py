#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run example/register_images.py, preprocess_tissues.py, and analyze_tissues.py
logic end-to-end for one .mat file.

Per-sample outputs are written next to the input file (same as path = dirname(fname)
in those scripts). The combined summary CSV is data/all_samples_output.csv relative
to the current working directory, exactly like analyze_tissues.py (run from the
example/ folder if your .mat files live in example/data/).

Constants match the example scripts; this pipeline assumes one tissue region
(grid / is_one_region=True) only.

Input file: data/example.mat next to this script (i.e. example/data/example.mat).

Usage (PYTHONPATH=src or editable install):
  python example/run_single_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time
import time as timer
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from skimage import io as skio


def run_register_single(fname: str) -> None:
    """example/register_images.py logic for one file."""
    p = Path(fname).resolve()
    sample = p.stem
    path = str(p.parent)

    imreg.NCORES = 10
    failed_reg = []
    registering_times = []
    print(f"Registering {fname}...")
    start = timer.time()
    try:
        data = imu.load_data(fname, videothresh=videothresh, fix_cut=False)
        warped_data, displacements = imreg.register_all_frames(data)
        io.savemat(f"{path}/{sample}_warped.mat", {"warped_data": warped_data})
        imu.save_data(f"{path}/{sample}_warped.tif", warped_data)
        imu.save_data(f"{path}/{sample}.tif", data)
    except Exception:
        failed_reg.append(fname)
        print(f"Error registering {fname}. Skipping this file.")
        raise SystemExit(1)
    registering_times.append(timer.time() - start)
    print(f"Registration completed for 1 files.")
    print(f"Total registration time: {sum(registering_times):.2f} seconds.")
    print(f"Average registration time per file: {np.mean(registering_times):.2f} seconds.")
    print(f"Failed registrations: {len(failed_reg)} files.")
    if failed_reg:
        print("Failed files:", failed_reg)


def run_preprocess_single(fname: str) -> None:
    """example/preprocess_tissues.py logic for one file."""
    p = Path(fname).resolve()
    sample = p.stem
    path = str(p.parent)

    imreg.NCORES = 10
    preprocessing_times = []
    print(f"Processing {sample}...")
    start = time.time()
    data = imu.load_data(fname, videothresh=videothresh)

    print("Creating tissue mask...")
    mask = imu.get_tissue_mask(data, interactive=False)
    skio.imsave(f"{path}/{sample}_tissue_mask.tif", mask.astype(np.uint8) * 255)

    print("Selecting regions")
    data_2d, mask_2d = imu.stack_first_frame_for_rotate(data, mask)
    data_2d, mask_2d = imu.rotate_data(data_2d, mask_2d)

    region_params = [tissue_div_x, tissue_div_y]

    print("Saving preprocessing parameters..")
    np.savez(
        f"{path}/{sample}_preprocessing.npz",
        mask=mask.squeeze(),
        region_params=np.array(region_params),
        type=np.array([True]),
    )
    preprocessing_times.append(time.time() - start)
    print("done.\n")
    print(f"Preprocessing completed for 1 files.")
    print(f"Total preprocessing time: {sum(preprocessing_times):.2f} seconds.")
    print(f"Average preprocessing time per file: {np.mean(preprocessing_times):.2f} seconds.")


def run_analyze_single(fname: str) -> None:
    """example/analyze_tissues.py logic for one file."""
    p = Path(fname).resolve()
    sample = p.stem
    path = str(p.parent)

    imreg.NCORES = 8

    print(f"Processing {sample}...")

    if not os.path.exists(f"{path}/{sample}_preprocessing.npz"):
        print("Preprocessing file not found. Please run the preprocessing script first.")
        return
    print("Loading preprocessing info...")
    preprocess_info = np.load(f"{path}/{sample}_preprocessing.npz")

    region_params = preprocess_info["region_params"]
    mask = preprocess_info["mask"]

    if not os.path.exists(f"{path}/{sample}_warped.mat"):
        print("Warped images not found. Please run the preprocessing script first.")
        return
    print("Loading warped images...")
    warped_data = io.loadmat(f"{path}/{sample}_warped.mat")["warped_data"]

    try:
        start = timer.time()

        print("Rotating data...")
        warped_data, mask = imu.rotate_data_cv2(warped_data, mask)
        first_frame = warped_data[0]
        skio.imsave(
            f"{path}/{sample}_tissue_mask_rotated.tif",
            mask.astype(np.uint8) * 255,
        )
        skio.imsave(
            f"{path}/{sample}_warped_tissue_rotated.tif",
            warped_data.astype(np.int16),
            check_contrast=False,
        )

        regions = imu.divide_tissue_in_regions(
            mask=mask, nx=region_params[0], ny=region_params[1]
        )

        print("Analyzing tissue intensities...")
        tissue_trace = imu.evaluate_regional_intensities(
            warped_data, mask.astype(int)
        )[:, 0]

        filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace(
            tissue_trace
        )

        (
            bpm,
            bpm_std,
            timing_irregularity,
            upstroke_time,
            amplitude,
        ) = ca.trace_outputs(
            filtered_trace, max_peaks_idx, min_peaks_idx, framerate
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
                trace, framerate=framerate, width_factor=width_factor
            )
            filtered_traces.append(filtered_trace)
            max_peaks.append(max_peaks_idx)
            min_peaks.append(min_peaks_idx)

            if len(max_peaks_idx) <= 2:
                continue
            bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = (
                ca.trace_outputs(
                    filtered_trace, max_peaks_idx, min_peaks_idx, framerate
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

        if plot_all_traces:
            ntraces = len(filtered_traces)
            fig, axs = plt.subplots(
                ntraces, 1, figsize=(10, 2 * ntraces), sharex=True
            )
            time_trace = np.arange(len(calcium_traces[0].trace)) / framerate
            for i, trace in enumerate(filtered_traces):
                max_peaks_idx = max_peaks[i]
                min_peaks_idx = min_peaks[i]

                axs[i].plot(time_trace, trace, "k", label="Filtered Trace")
                if len(max_peaks_idx) > 0:
                    axs[i].plot(
                        max_peaks_idx / framerate,
                        trace[max_peaks_idx],
                        "ro",
                        label="Max Peaks",
                    )
                if len(min_peaks_idx) > 0:
                    axs[i].plot(
                        min_peaks_idx / framerate,
                        trace[min_peaks_idx],
                        "bo",
                        label="Min Peaks",
                    )
                axs[i].set_ylabel("Intensity")

            plt.tight_layout()
            plt.savefig(
                f"{path}/{sample}_all_individual_traces.png",
                dpi=180,
                bbox_inches="tight",
            )

        fig1, fig2 = pu.plot_regions_traces(
            first_frame, regions, filtered_traces, framerate=framerate
        )
        fig1.savefig(f"{path}/{sample}_all_regions.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"{path}/{sample}_all_traces.png", dpi=300, bbox_inches="tight")

        filtered_traces = np.array(filtered_traces)
        synchronicity = np.mean(np.corrcoef(filtered_traces))
        print(f"Synchronicity: {synchronicity:.2f}")

        tissue_header = [
            "Sample Name",
            "bpm",
            "bpm std",
            "timing irreg",
            "synchronicity",
            "upstroke time",
            "amplitude",
        ]
        fields = np.array(
            [
                [
                    sample,
                    tissue_calcium_trace.bpm,
                    tissue_calcium_trace.bpm_std,
                    tissue_calcium_trace.timing_irregularity,
                    synchronicity,
                    tissue_calcium_trace.upstroke_time,
                    tissue_calcium_trace.amplitude,
                ]
            ]
        )

        np.savetxt(
            f"{path}/{sample}_output.csv",
            fields,
            delimiter=",",
            fmt="%s",
            header=",".join(tissue_header),
            comments="",
        )

        np.savetxt(
            f"{selected_folder}/all_samples_output.csv",
            fields,
            delimiter=",",
            fmt="%s",
            header=",".join(tissue_header),
            comments="",
        )

        valid_mask = np.isin(regions, valid_regions)
        regions[~valid_mask] = 0
        aux_regions = np.copy(regions)
        for i, ctrace in enumerate(calcium_traces):
            regions[aux_regions == ctrace.region] = i + 1
            ctrace.region = i + 1

        region_header = [
            "Sample Name",
            "Region",
            "bpm",
            "bpm std",
            "timing irreg",
            "synchronicity",
            "upstroke time",
            "amplitude",
        ]
        outputs = []
        for i, ctrace in enumerate(calcium_traces):
            outputs.append(
                [
                    sample,
                    ctrace.region,
                    ctrace.bpm,
                    ctrace.bpm_std,
                    ctrace.timing_irregularity,
                    synchronicity,
                    ctrace.upstroke_time,
                    ctrace.amplitude,
                ]
            )

        np.savetxt(
            f"{path}/{sample}_region_output.csv",
            np.array(outputs),
            delimiter=",",
            fmt="%s",
            header=",".join(region_header),
            comments="",
        )

        clean_traces = np.array([t.trace for t in calcium_traces])
        time = np.arange(len(tissue_calcium_trace.trace)) / framerate
        traces_raw = [time]
        traces_raw += [tissue_calcium_trace.trace]
        traces_raw += [trace.trace for trace in calcium_traces]
        traces_raw = np.vstack(traces_raw).T

        np.savetxt(
            f"{path}/{sample}_raw_output.csv",
            traces_raw,
            delimiter=",",
            fmt="%s",
            header="Time,"
            + ",".join(
                ["Tissue"] + [f"Region {i+1}" for i in range(len(calcium_traces))]
            ),
            comments="",
        )

        fig1, fig2 = pu.plot_regions_traces(
            first_frame, regions, clean_traces, framerate=framerate
        )
        fig1.savefig(
            f"{path}/{sample}_regular_regions.png", dpi=300, bbox_inches="tight"
        )
        fig2.savefig(
            f"{path}/{sample}_regular_traces.png", dpi=300, bbox_inches="tight"
        )
        plt.close("all")

        elapsed = timer.time() - start
        print("done.\n")
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return

    print("Processing completed for 1 files.")
    print(f"Total processing time: {elapsed:.2f} seconds.")
    print(f"Average processing time per file: {elapsed:.2f} seconds.")
    print("Failed analyses: 0 files.")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import imregistration as imreg
import imutils as imu
import calcium_analysis as ca
import plotutils as pu

# USER INPUTS — example/register_images.py
videothresh = (100, 101)
# force_registration = False  # defined in register_images.py but unused

# USER INPUTS — example/preprocess_tissues.py
framerate = 65.18
pixelsize = 0.908
tissue_div_x = 2
tissue_div_y = 6
threshold_value = 0  # If 0, otsu thresholding is used (unused in script body)
# force_preprocessing = True  # defined in preprocess_tissues.py but unused

# USER INPUTS — example/analyze_tissues.py
# videothresh / pixelsize / tissue_div_x / tissue_div_y duplicated there for reference
# threshold_value duplicated
width_factor = 1 / 10
plot_all_traces = True
selected_folder = "data/"

# Fixed input path (example/data/example.mat)
INPUT_MAT = Path(__file__).resolve().parent / "data" / "example.mat"

mat_path = INPUT_MAT.resolve()
if not mat_path.is_file():
    raise SystemExit(f"Not found: {mat_path}")
fname = str(mat_path)
if "warped" in fname:
    raise SystemExit("Pick the raw .mat, not a *_warped.mat file.")

run_register_single(fname)
# run_preprocess_single(fname)
# run_analyze_single(fname)
