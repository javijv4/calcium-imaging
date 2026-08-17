# PySide6 Desktop GUI for Calcium Imaging

## Summary
Build a Python desktop GUI on a new branch `feature/pyside6-gui`, with all GUI-related code under `src/` in a new `src/calcium_gui/` package. The app will wrap the existing Python pipeline, let users load a folder of samples, run registration/preprocess/analysis on one sample or the full folder, configure pipeline options from the top menu bar, and inspect outputs in a left sample tree plus a right-side viewer with a time slider for stack data.

## Key Changes
- Create branch `feature/pyside6-gui` from the current clean `main` branch before any code changes.
- Add `PySide6` as a runtime dependency and expose a GUI entrypoint such as `calcium-imaging-gui` plus `python -m calcium_gui`.
- Keep existing processing modules in `src/` as the backend, and add a thin orchestration layer so the GUI calls Python functions directly instead of shelling out to the current scripts.
- Refactor the script-level workflows in `register_images.py`, `preprocess_tissues.py`, and `analyze_tissues.py` into reusable pipeline functions that accept explicit parameters and return artifact paths, metrics, and status without opening Tk windows.
- Add `src/calcium_gui/` with these responsibilities:
  - `main.py` / app bootstrap and window startup
  - `main_window.py` for the menu bar, left sample panel, right viewer, log/progress area, and action wiring
  - `models.py` for sample records, artifact records, and GUI settings/state
  - `pipeline.py` for sample discovery, output-file grouping, and single/batch pipeline execution
  - `workers.py` for background execution so registration/analysis do not freeze the UI
  - `viewer.py` for image/stack/plot/table viewing logic
  - `dialogs.py` for parameter editing and interactive preprocessing dialogs
- Use a `QTreeWidget` in the left panel:
  - Top level: one node per sample
  - Children: input file and discovered/generated outputs such as warped stack, masks, PNG results, CSV outputs, preprocessing files
  - Support collapse/expand and refresh after each run
- Use a right-side stacked viewer:
  - 2D image files (`.png`, `.tif`) shown as images
  - stack-like files (`.mat`, `.tif/.tiff`, `.nd2`, `.czi`, warped outputs) loaded through existing IO helpers and displayed with a frame slider
  - CSV outputs shown in a table view
  - analysis PNG outputs shown directly so result figures are inspectable from the tree
- Put all pipeline options in the top menu bar:
  - `File`: open folder, refresh, quit
  - `Pipeline`: run registration, preprocess, analysis, or full pipeline for selected sample or all samples
  - `Settings`: open dialogs/tabs for general IO, registration, preprocess, and analysis parameters
  - `View`: reset viewer, toggle overlays/logs if needed
- Replace current ad hoc interactive Tk/Matplotlib prompts with GUI-owned dialogs:
  - folder selection handled by the main app
  - time-range selection exposed as a dialog with a trace plot and range controls
  - preprocessing mode and threshold controls exposed as Qt dialogs instead of standalone Tk windows
  - keep the first version focused on the current interactions already present in the pipeline; do not add new manual segmentation tools beyond parity with existing behavior
- Preserve compatibility with the current scripts by having them call the new reusable pipeline functions, so GUI and scripts share one execution path.

## Test Plan
- Keep existing pipeline regression tests passing unchanged.
- Add unit tests for sample discovery and artifact grouping from a folder structure with mixed inputs/outputs.
- Add unit tests for pipeline parameter objects and defaults, including single-sample vs batch execution selection.
- Add focused GUI tests with `pytest-qt` for:
  - loading a folder populates the sample tree correctly
  - selecting a tree item chooses the expected viewer mode
  - long-running actions dispatch to a worker instead of blocking the main thread
  - completing a pipeline step refreshes the left panel with new artifacts
- Manually verify on example data:
  - open a folder with sample files
  - run each step on one sample
  - run full-folder processing
  - open warped stacks and move through time with the slider
  - open masks/PNGs/CSVs from the left panel and confirm they render appropriately

## Assumptions
- `PySide6` is the chosen GUI toolkit.
- All GUI code may live under `src/` in `src/calcium_gui/`; non-GUI backend modules stay where they are.
- First version targets a local desktop workflow, not a FIJI plugin and not a browser app.
- First version will support the current interactive preprocessing decisions inside the GUI, but will not introduce new custom annotation/editing tools beyond existing pipeline behavior.
- Default branch name will be `feature/pyside6-gui` unless you want a different naming convention.
