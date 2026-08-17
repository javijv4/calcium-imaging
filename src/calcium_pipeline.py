#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared pipeline services for CLI scripts and the desktop GUI."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scipy_io
from skimage import filters
from skimage import io as skio

import calcium_analysis as ca
import imio
import imregistration as imreg
import imutils as imu
import plotutils as pu

SUPPORTED_INPUT_EXTENSIONS = (".mat", ".tif", ".tiff", ".nd2", ".czi")

ARTIFACT_SPECS = (
    ("_tissue_mask_rotated.tif", "rotated_mask", "Rotated tissue mask"),
    ("_warped_tissue_rotated.tif", "rotated_stack", "Rotated warped stack"),
    ("_all_individual_traces.png", "individual_traces_png", "Individual traces"),
    ("_regular_regions.png", "regular_regions_png", "Regular regions"),
    ("_regular_traces.png", "regular_traces_png", "Regular traces"),
    ("_region_output.csv", "region_csv", "Region summary"),
    ("_raw_output.csv", "raw_csv", "Raw traces"),
    ("_preprocessing.npz", "preprocess", "Preprocessing parameters"),
    ("_all_regions.png", "regions_png", "All regions"),
    ("_all_traces.png", "traces_png", "All traces"),
    ("_tissue_mask.tif", "mask", "Tissue mask"),
    ("_warped.mat", "warped_stack", "Warped stack (.mat)"),
    ("_warped.tif", "warped_tiff", "Warped stack (.tif)"),
    ("_output.csv", "summary_csv", "Sample summary"),
)

ARTIFACT_PRIORITY = {
    "input": 0,
    "alternate_input": 1,
    "warped_stack": 2,
    "warped_tiff": 3,
    "mask": 4,
    "rotated_mask": 5,
    "rotated_stack": 6,
    "preprocess": 7,
    "summary_csv": 8,
    "region_csv": 9,
    "raw_csv": 10,
    "regions_png": 11,
    "traces_png": 12,
    "individual_traces_png": 13,
    "regular_regions_png": 14,
    "regular_traces_png": 15,
}

INPUT_PRIORITY = {
    ".mat": 0,
    ".nd2": 1,
    ".czi": 2,
    ".tif": 3,
    ".tiff": 4,
}

PipelineLogger = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, str], None]


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    label: str
    path: Path


@dataclass(frozen=True)
class SampleRecord:
    name: str
    folder: Path
    input_path: Path
    artifacts: tuple[ArtifactRecord, ...] = ()


@dataclass
class PipelineSettings:
    framerate: float = 65.18
    videothresh_start: int | None = 100
    videothresh_end: int | None = 600
    fix_cut: bool = True
    scene: int = 0
    channel: int = 0
    z_index: int = 0
    registration_cores: int = 10
    analysis_cores: int = 8
    tissue_div_x: int = 2
    tissue_div_y: int = 6
    force_registration: bool = False
    force_preprocessing: bool = False
    region_mode: str = "grid"
    threshold_value: float | None = None
    width_factor: float = 1 / 10
    plot_all_traces: bool = True

    def videothresh(self) -> tuple[int, int] | None:
        if self.videothresh_start is None or self.videothresh_end is None:
            return None
        if self.videothresh_end <= self.videothresh_start:
            raise ValueError("videothresh_end must be greater than videothresh_start")
        return (self.videothresh_start, self.videothresh_end)


@dataclass
class SampleRunResult:
    sample: str
    step: str
    status: str
    outputs: tuple[Path, ...] = ()
    duration_seconds: float = 0.0
    summary_row: tuple[object, ...] | None = None
    error: str | None = None


def _emit(log: PipelineLogger | None, message: str) -> None:
    if log is not None:
        log(message)


def _write_csv_rows(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _input_sort_key(path: Path) -> tuple[int, str]:
    return (INPUT_PRIORITY.get(path.suffix.lower(), 99), path.name.lower())


def _artifact_sort_key(artifact: ArtifactRecord) -> tuple[int, str]:
    return (ARTIFACT_PRIORITY.get(artifact.kind, 99), artifact.path.name.lower())


def _strip_artifact_suffix(name: str) -> str | None:
    for suffix, _, _ in ARTIFACT_SPECS:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def infer_sample_name(path: str | Path) -> str | None:
    candidate = Path(path)
    stripped = _strip_artifact_suffix(candidate.name)
    if stripped is not None:
        return stripped
    if candidate.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        return None
    if candidate.stem.endswith("_warped"):
        return candidate.stem[: -len("_warped")]
    return candidate.stem


def is_primary_input_candidate(path: str | Path) -> bool:
    candidate = Path(path)
    if candidate.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        return False
    if _strip_artifact_suffix(candidate.name) is not None:
        return False
    if candidate.stem.endswith("_warped"):
        return False
    return True


def _artifact_record_for_path(path: Path, primary_input: Path) -> ArtifactRecord | None:
    if path == primary_input:
        return ArtifactRecord("input", f"Input ({path.suffix.lower()})", path)

    for suffix, kind, label in ARTIFACT_SPECS:
        if path.name.endswith(suffix):
            return ArtifactRecord(kind, label, path)

    if path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS and path != primary_input:
        return ArtifactRecord("alternate_input", f"Alternate input ({path.suffix.lower()})", path)

    return None


def discover_samples(folder: str | Path) -> list[SampleRecord]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    grouped_inputs: dict[str, list[Path]] = {}
    grouped_paths: dict[str, list[Path]] = {}
    for path in folder_path.iterdir():
        if not path.is_file():
            continue
        sample_name = infer_sample_name(path)
        if sample_name is None:
            continue
        grouped_paths.setdefault(sample_name, []).append(path)
        if is_primary_input_candidate(path):
            grouped_inputs.setdefault(sample_name, []).append(path)

    samples: list[SampleRecord] = []
    for sample_name, input_paths in grouped_inputs.items():
        sorted_inputs = sorted(input_paths, key=_input_sort_key)
        primary_input = sorted_inputs[0]
        artifacts = [
            artifact
            for artifact in (
                _artifact_record_for_path(path, primary_input)
                for path in grouped_paths.get(sample_name, [])
            )
            if artifact is not None
        ]
        artifacts.sort(key=_artifact_sort_key)
        samples.append(
            SampleRecord(
                name=sample_name,
                folder=folder_path,
                input_path=primary_input,
                artifacts=tuple(artifacts),
            )
        )

    return sorted(samples, key=lambda sample: sample.name.lower())


def find_sample(folder: str | Path, sample_name: str) -> SampleRecord:
    for sample in discover_samples(folder):
        if sample.name == sample_name:
            return sample
    raise FileNotFoundError(f"Sample {sample_name!r} not found in {folder}")


def load_stack_for_viewer(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".mat":
        mat = scipy_io.loadmat(source)
        if "warped_data" in mat:
            arr = mat["warped_data"]
        else:
            arr = imio.load_mat(str(source))
    elif source.suffix.lower() in (".tif", ".tiff", ".nd2", ".czi"):
        arr = imio.load_image(str(source))
    else:
        raise ValueError(f"Unsupported stack format: {source.suffix}")

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return np.asarray(arr)


def _load_data(input_path: Path, settings: PipelineSettings) -> np.ndarray:
    return imu.load_data(
        str(input_path),
        videothresh=settings.videothresh(),
        fix_cut=settings.fix_cut,
        scene=settings.scene,
        channel=settings.channel,
        z=settings.z_index,
    )


def _registration_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_warped.mat")


def _preprocess_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_preprocessing.npz")


def _mask_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_tissue_mask.tif")


def run_registration(
    input_path: str | Path,
    settings: PipelineSettings,
    log: PipelineLogger | None = None,
) -> SampleRunResult:
    sample_path = Path(input_path)
    sample_name = sample_path.stem
    output_path = _registration_output_path(sample_path)

    if output_path.exists() and not settings.force_registration:
        _emit(log, f"{sample_name}: registration already exists, skipping.")
        return SampleRunResult(sample_name, "registration", "skipped", outputs=(output_path,))

    start = time.time()
    _emit(log, f"{sample_name}: loading data for registration.")
    data = _load_data(sample_path, settings)
    imreg.NCORES = settings.registration_cores
    _emit(log, f"{sample_name}: running registration.")
    warped_data, _ = imreg.register_all_frames(data)
    scipy_io.savemat(output_path, {"warped_data": warped_data})
    duration = time.time() - start
    _emit(log, f"{sample_name}: registration finished in {duration:.2f}s.")
    return SampleRunResult(
        sample_name,
        "registration",
        "completed",
        outputs=(output_path,),
        duration_seconds=duration,
    )


def run_preprocess(
    input_path: str | Path,
    settings: PipelineSettings,
    log: PipelineLogger | None = None,
) -> SampleRunResult:
    sample_path = Path(input_path)
    sample_name = sample_path.stem
    preprocess_path = _preprocess_output_path(sample_path)
    mask_path = _mask_output_path(sample_path)

    if preprocess_path.exists() and not settings.force_preprocessing:
        _emit(log, f"{sample_name}: preprocessing already exists, skipping.")
        outputs = tuple(path for path in (mask_path, preprocess_path) if path.exists())
        return SampleRunResult(sample_name, "preprocess", "skipped", outputs=outputs)

    start = time.time()
    _emit(log, f"{sample_name}: loading data for preprocessing.")
    data = _load_data(sample_path, settings)
    _emit(log, f"{sample_name}: creating tissue mask.")
    mask = imu.get_tissue_mask(data, interactive=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255)

    data_2d, mask_2d = imu.stack_first_frame_for_rotate(data, mask)
    data_2d, mask_2d = imu.rotate_data(data_2d, mask_2d)
    if settings.region_mode == "grid":
        is_one_region = True
        region_params = np.array([settings.tissue_div_x, settings.tissue_div_y], dtype=float)
    elif settings.region_mode == "threshold":
        threshold_value = settings.threshold_value
        if threshold_value is None:
            source = imu.max_over_time(data_2d) if data_2d.ndim == 3 else data_2d
            threshold_value = float(filters.threshold_otsu(source[mask_2d > 0]))
        is_one_region = False
        region_params = np.array([threshold_value, 0], dtype=float)
    else:
        raise ValueError(f"Unsupported region mode: {settings.region_mode}")

    np.savez(
        preprocess_path,
        mask=mask.squeeze(),
        region_params=region_params,
        type=np.array([is_one_region]),
    )
    duration = time.time() - start
    _emit(log, f"{sample_name}: preprocessing finished in {duration:.2f}s.")
    return SampleRunResult(
        sample_name,
        "preprocess",
        "completed",
        outputs=(mask_path, preprocess_path),
        duration_seconds=duration,
    )


def _build_summary_row(
    sample_name: str,
    tissue_trace: ca.CalciumTrace | None,
    synchronicity: float,
) -> tuple[object, ...]:
    if tissue_trace is None:
        return (sample_name, "", "", "", synchronicity, "", "")
    return (
        sample_name,
        tissue_trace.bpm,
        tissue_trace.bpm_std,
        tissue_trace.timing_irregularity,
        synchronicity,
        tissue_trace.upstroke_time,
        tissue_trace.amplitude,
    )


def run_analysis(
    input_path: str | Path,
    settings: PipelineSettings,
    log: PipelineLogger | None = None,
) -> SampleRunResult:
    sample_path = Path(input_path)
    sample_name = sample_path.stem
    preprocess_path = _preprocess_output_path(sample_path)
    warped_path = _registration_output_path(sample_path)
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Missing preprocessing file for {sample_name}: {preprocess_path.name}")
    if not warped_path.exists():
        raise FileNotFoundError(f"Missing warped file for {sample_name}: {warped_path.name}")

    start = time.time()
    _emit(log, f"{sample_name}: loading preprocessing outputs.")
    preprocess_info = np.load(preprocess_path)
    is_one_region = bool(preprocess_info["type"][0])
    region_params = preprocess_info["region_params"]
    mask = preprocess_info["mask"]

    imreg.NCORES = settings.analysis_cores
    warped_data = scipy_io.loadmat(warped_path)["warped_data"]
    warped_data, mask = imu.rotate_data_cv2(warped_data, mask)
    first_frame = warped_data[0]

    rotated_mask_path = sample_path.with_name(f"{sample_name}_tissue_mask_rotated.tif")
    rotated_stack_path = sample_path.with_name(f"{sample_name}_warped_tissue_rotated.tif")
    skio.imsave(rotated_mask_path, mask.astype(np.uint8) * 255)
    skio.imsave(rotated_stack_path, warped_data.astype(np.int16), check_contrast=False)

    if is_one_region:
        regions = imu.divide_tissue_in_regions(mask=mask, nx=int(region_params[0]), ny=int(region_params[1]))
    else:
        regions = imu.apply_threshold(final_threshold=float(region_params[0]), data=first_frame, tissue_mask=mask)

    tissue_calcium_trace = None
    if is_one_region:
        _emit(log, f"{sample_name}: analyzing tissue-level trace.")
        tissue_trace = imu.evaluate_regional_intensities(warped_data, mask.astype(int))[:, 0]
        filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace(tissue_trace)
        if len(max_peaks_idx) <= 2:
            bpm = bpm_std = timing_irregularity = upstroke_time = amplitude = 0
        else:
            bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(
                filtered_trace, max_peaks_idx, min_peaks_idx, settings.framerate
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

    _emit(log, f"{sample_name}: analyzing regional traces.")
    traces = imu.evaluate_regional_intensities(warped_data, regions)
    calcium_traces = []
    valid_regions = []
    filtered_traces = []
    max_peaks_per_region = []
    min_peaks_per_region = []
    for index, trace in enumerate(traces.T):
        filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace_fft(
            trace,
            framerate=settings.framerate,
            width_factor=settings.width_factor,
        )
        filtered_traces.append(filtered_trace)
        max_peaks_per_region.append(max_peaks_idx)
        min_peaks_per_region.append(min_peaks_idx)
        if len(max_peaks_idx) <= 2:
            continue
        bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(
            filtered_trace,
            max_peaks_idx,
            min_peaks_idx,
            settings.framerate,
        )
        ctrace = ca.CalciumTrace(
            filtered_trace,
            max_peaks_idx,
            min_peaks_idx,
            index + 1,
            bpm,
            bpm_std,
            timing_irregularity,
            upstroke_time,
            amplitude,
        )
        calcium_traces.append(ctrace)
        valid_regions.append(ctrace.region)

    output_paths: list[Path] = [rotated_mask_path, rotated_stack_path]

    if settings.plot_all_traces and filtered_traces:
        fig, axs = plt.subplots(len(filtered_traces), 1, figsize=(10, 2 * len(filtered_traces)), sharex=True)
        if len(filtered_traces) == 1:
            axs = [axs]
        time_trace = np.arange(len(filtered_traces[0])) / settings.framerate
        for axis, trace, max_peaks_idx, min_peaks_idx in zip(
            axs, filtered_traces, max_peaks_per_region, min_peaks_per_region
        ):
            axis.plot(time_trace, trace, "k")
            if len(max_peaks_idx) > 0:
                axis.plot(max_peaks_idx / settings.framerate, trace[max_peaks_idx], "ro")
            if len(min_peaks_idx) > 0:
                axis.plot(min_peaks_idx / settings.framerate, trace[min_peaks_idx], "bo")
            axis.set_ylabel("Intensity")
        plt.tight_layout()
        all_individual_path = sample_path.with_name(f"{sample_name}_all_individual_traces.png")
        plt.savefig(all_individual_path, dpi=180, bbox_inches="tight")
        output_paths.append(all_individual_path)

    fig1, fig2 = pu.plot_regions_traces(first_frame, regions, filtered_traces, framerate=settings.framerate)
    all_regions_path = sample_path.with_name(f"{sample_name}_all_regions.png")
    all_traces_path = sample_path.with_name(f"{sample_name}_all_traces.png")
    fig1.savefig(all_regions_path, dpi=300, bbox_inches="tight")
    fig2.savefig(all_traces_path, dpi=300, bbox_inches="tight")
    output_paths.extend([all_regions_path, all_traces_path])

    filtered_trace_array = np.array(filtered_traces)
    if filtered_trace_array.size == 0:
        synchronicity = 0.0
    elif filtered_trace_array.ndim == 1:
        synchronicity = 1.0
    else:
        synchronicity = float(np.mean(np.corrcoef(filtered_trace_array)))

    summary_header = [
        "Sample Name",
        "bpm",
        "bpm std",
        "timing irreg",
        "synchronicity",
        "upstroke time",
        "amplitude",
    ]
    summary_row = _build_summary_row(sample_name, tissue_calcium_trace, synchronicity)
    summary_path = sample_path.with_name(f"{sample_name}_output.csv")
    _write_csv_rows(summary_path, summary_header, [summary_row])
    output_paths.append(summary_path)

    valid_mask = np.isin(regions, valid_regions)
    regions = np.array(regions, copy=True)
    regions[~valid_mask] = 0
    aux_regions = np.copy(regions)
    for index, ctrace in enumerate(calcium_traces):
        regions[aux_regions == ctrace.region] = index + 1
        ctrace.region = index + 1

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
    region_rows = [
        (
            sample_name,
            ctrace.region,
            ctrace.bpm,
            ctrace.bpm_std,
            ctrace.timing_irregularity,
            synchronicity,
            ctrace.upstroke_time,
            ctrace.amplitude,
        )
        for ctrace in calcium_traces
    ]
    region_output_path = sample_path.with_name(f"{sample_name}_region_output.csv")
    _write_csv_rows(region_output_path, region_header, region_rows)
    output_paths.append(region_output_path)

    time_axis = np.arange(warped_data.shape[0]) / settings.framerate
    tissue_trace_row = tissue_calcium_trace.trace if tissue_calcium_trace is not None else np.zeros(warped_data.shape[0])
    raw_rows = np.vstack([time_axis, tissue_trace_row, *[trace.trace for trace in calcium_traces]]).T
    raw_header = ["Time", "Tissue"] + [f"Region {index + 1}" for index in range(len(calcium_traces))]
    raw_output_path = sample_path.with_name(f"{sample_name}_raw_output.csv")
    _write_csv_rows(raw_output_path, raw_header, raw_rows)
    output_paths.append(raw_output_path)

    clean_traces = np.array([trace.trace for trace in calcium_traces])
    fig1, fig2 = pu.plot_regions_traces(first_frame, regions, clean_traces, framerate=settings.framerate)
    regular_regions_path = sample_path.with_name(f"{sample_name}_regular_regions.png")
    regular_traces_path = sample_path.with_name(f"{sample_name}_regular_traces.png")
    fig1.savefig(regular_regions_path, dpi=300, bbox_inches="tight")
    fig2.savefig(regular_traces_path, dpi=300, bbox_inches="tight")
    output_paths.extend([regular_regions_path, regular_traces_path])
    plt.close("all")

    duration = time.time() - start
    _emit(log, f"{sample_name}: analysis finished in {duration:.2f}s.")
    return SampleRunResult(
        sample_name,
        "analysis",
        "completed",
        outputs=tuple(output_paths),
        duration_seconds=duration,
        summary_row=summary_row,
    )


def run_pipeline_for_sample(
    input_path: str | Path,
    step: str,
    settings: PipelineSettings,
    log: PipelineLogger | None = None,
) -> SampleRunResult:
    sample_name = Path(input_path).stem
    try:
        if step == "registration":
            return run_registration(input_path, settings, log=log)
        if step == "preprocess":
            return run_preprocess(input_path, settings, log=log)
        if step == "analysis":
            return run_analysis(input_path, settings, log=log)
        if step == "full":
            registration_result = run_registration(input_path, settings, log=log)
            preprocess_result = run_preprocess(input_path, settings, log=log)
            analysis_result = run_analysis(input_path, settings, log=log)
            return SampleRunResult(
                sample=analysis_result.sample,
                step=step,
                status=analysis_result.status,
                outputs=registration_result.outputs + preprocess_result.outputs + analysis_result.outputs,
                duration_seconds=(
                    registration_result.duration_seconds
                    + preprocess_result.duration_seconds
                    + analysis_result.duration_seconds
                ),
                summary_row=analysis_result.summary_row,
            )
        raise ValueError(f"Unsupported pipeline step: {step}")
    except Exception as exc:  # noqa: BLE001 - keep batch processing alive.
        _emit(log, f"{sample_name}: {step} failed: {exc}")
        return SampleRunResult(sample_name, step, "failed", error=str(exc))


def run_pipeline_for_folder(
    folder: str | Path,
    step: str,
    settings: PipelineSettings,
    sample_names: Iterable[str] | None = None,
    log: PipelineLogger | None = None,
    progress: ProgressCallback | None = None,
) -> list[SampleRunResult]:
    selected_names = set(sample_names) if sample_names is not None else None
    samples = [
        sample
        for sample in discover_samples(folder)
        if selected_names is None or sample.name in selected_names
    ]
    results: list[SampleRunResult] = []
    summary_rows = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        if progress is not None:
            progress(index - 1, total, sample.name, "started")
        result = run_pipeline_for_sample(sample.input_path, step, settings, log=log)
        results.append(result)
        if result.summary_row is not None:
            summary_rows.append(result.summary_row)
        if progress is not None:
            progress(index, total, sample.name, result.status)

    if step in {"analysis", "full"} and summary_rows:
        summary_path = Path(folder) / "all_samples_output.csv"
        _write_csv_rows(
            summary_path,
            [
                "Sample Name",
                "bpm",
                "bpm std",
                "timing irreg",
                "synchronicity",
                "upstroke time",
                "amplitude",
            ],
            summary_rows,
        )

    return results


def summarize_results(results: Iterable[SampleRunResult]) -> tuple[int, int, int]:
    completed = skipped = failed = 0
    for result in results:
        if result.status == "completed":
            completed += 1
        elif result.status == "skipped":
            skipped += 1
        else:
            failed += 1
    return completed, skipped, failed
