"""Background workers for long-running pipeline steps."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from calcium_pipeline import PipelineSettings, run_pipeline_for_folder


class PipelineWorker(QObject):
    """Run a folder pipeline action off the GUI thread."""

    log_message = Signal(str)
    progress_update = Signal(int, int, str, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, step: str, settings: PipelineSettings, sample_names=None):
        super().__init__()
        self.folder = folder
        self.step = step
        self.settings = PipelineSettings(**settings.__dict__)
        self.sample_names = list(sample_names) if sample_names is not None else None

    @Slot()
    def run(self) -> None:
        try:
            results = run_pipeline_for_folder(
                self.folder,
                self.step,
                self.settings,
                sample_names=self.sample_names,
                log=self.log_message.emit,
                progress=self.progress_update.emit,
            )
            self.finished.emit(results)
        except Exception as exc:  # noqa: BLE001 - surface unexpected worker failures.
            self.failed.emit(str(exc))

