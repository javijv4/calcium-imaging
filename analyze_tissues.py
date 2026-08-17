#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze all preprocessed samples in a selected folder."""

from __future__ import annotations

from calcium_pipeline import PipelineSettings, run_pipeline_for_folder, summarize_results
from gui_utils import select_folder


def main() -> int:
    selected_folder = select_folder()
    if not selected_folder:
        print("No folder selected. Exiting.")
        return 0

    settings = PipelineSettings(
        framerate=65.18,
        videothresh_start=100,
        videothresh_end=600,
        tissue_div_x=2,
        tissue_div_y=6,
        analysis_cores=8,
        width_factor=1 / 10,
        plot_all_traces=True,
    )
    results = run_pipeline_for_folder(selected_folder, "analysis", settings, log=print)
    completed, skipped, failed = summarize_results(results)
    print(f"Analysis completed: {completed} completed, {skipped} skipped, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
