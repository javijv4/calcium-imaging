"""Main application window for the calcium-imaging GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QSplitter, QStatusBar, QTreeWidget, QTreeWidgetItem

from calcium_pipeline import PipelineSettings, discover_samples, summarize_results

from calcium_gui.dialogs import SettingsDialog
from calcium_gui.models import AppState
from calcium_gui.viewer import FileViewer
from calcium_gui.workers import PipelineWorker


class MainWindow(QMainWindow):
    """Top-level window containing the sample browser and viewer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calcium Imaging GUI")
        self.resize(1400, 900)

        self.state = AppState()
        self.settings = PipelineSettings()
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self._build_ui()
        self._build_menus()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)

        self.tree = QTreeWidget(splitter)
        self.tree.setHeaderLabel("Samples")
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.tree)

        right_splitter = QSplitter(Qt.Vertical, splitter)
        self.viewer = FileViewer(right_splitter)
        self.log_view = QPlainTextEdit(right_splitter)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        right_splitter.addWidget(self.viewer)
        right_splitter.addWidget(self.log_view)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 2)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

        status_bar = QStatusBar(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(220)
        status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(status_bar)
        self.statusBar().showMessage("Open a folder to begin.")

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Open Folder...", self.open_folder)
        file_menu.addAction("Refresh", self.refresh_samples)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        pipeline_menu = menu_bar.addMenu("Pipeline")
        self._add_pipeline_actions(pipeline_menu)

        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("Pipeline Options...", self.edit_settings)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Clear Viewer", self.viewer.clear)

    def _add_pipeline_actions(self, menu: QMenu) -> None:
        menu.addAction("Run Registration on Selected Sample", lambda: self.run_pipeline("registration", selected_only=True))
        menu.addAction("Run Preprocess on Selected Sample", lambda: self.run_pipeline("preprocess", selected_only=True))
        menu.addAction("Run Analysis on Selected Sample", lambda: self.run_pipeline("analysis", selected_only=True))
        menu.addAction("Run Full Pipeline on Selected Sample", lambda: self.run_pipeline("full", selected_only=True))
        menu.addSeparator()
        menu.addAction("Run Registration on Folder", lambda: self.run_pipeline("registration", selected_only=False))
        menu.addAction("Run Preprocess on Folder", lambda: self.run_pipeline("preprocess", selected_only=False))
        menu.addAction("Run Analysis on Folder", lambda: self.run_pipeline("analysis", selected_only=False))
        menu.addAction("Run Full Pipeline on Folder", lambda: self.run_pipeline("full", selected_only=False))

    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", str(self.state.folder or Path.cwd()))
        if not folder:
            return
        self.state.folder = Path(folder)
        self.refresh_samples()

    def refresh_samples(self) -> None:
        self.tree.clear()
        self.viewer.clear()
        if self.state.folder is None:
            return
        samples = discover_samples(self.state.folder)
        for sample in samples:
            sample_item = QTreeWidgetItem([sample.name])
            sample_item.setData(0, Qt.UserRole, {"sample": sample.name, "path": str(sample.input_path), "is_sample": True})
            self.tree.addTopLevelItem(sample_item)
            for artifact in sample.artifacts:
                child = QTreeWidgetItem([f"{artifact.label}: {artifact.path.name}"])
                child.setData(0, Qt.UserRole, {"sample": sample.name, "path": str(artifact.path), "is_sample": False})
                sample_item.addChild(child)
            sample_item.setExpanded(True)
        self.statusBar().showMessage(f"Loaded {len(samples)} samples from {self.state.folder}.")

    def edit_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.settings()
            self.append_log("Updated pipeline settings.")

    def selected_sample_name(self) -> str | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        payload = items[0].data(0, Qt.UserRole) or {}
        return payload.get("sample")

    def _on_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, Qt.UserRole) or {}
        path = payload.get("path")
        sample_name = payload.get("sample")
        if sample_name:
            self.state.selected_sample = sample_name
        if path:
            self.state.selected_path = Path(path)
            self.viewer.show_path(path)

    def run_pipeline(self, step: str, selected_only: bool) -> None:
        if self.state.folder is None:
            QMessageBox.information(self, "No Folder", "Open a folder before running the pipeline.")
            return
        sample_names = None
        if selected_only:
            sample_name = self.selected_sample_name()
            if sample_name is None:
                QMessageBox.information(self, "No Sample", "Select a sample or one of its files first.")
                return
            sample_names = [sample_name]
        self._start_worker(step, sample_names)

    def _start_worker(self, step: str, sample_names: list[str] | None) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Busy", "A pipeline task is already running.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage(f"Running {step}...")
        scope = ", ".join(sample_names) if sample_names else "folder"
        self.append_log(f"Starting {step} for {scope}.")

        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(self.state.folder, step, self.settings, sample_names=sample_names)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log)
        self.worker.progress_update.connect(self._on_progress_update)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _on_progress_update(self, current: int, total: int, sample_name: str, status: str) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"{status.capitalize()}: {sample_name} ({current}/{total})")

    def _on_worker_finished(self, results) -> None:
        completed, skipped, failed = summarize_results(results)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Finished. {completed} completed, {skipped} skipped, {failed} failed.")
        self.append_log(f"Finished run: {completed} completed, {skipped} skipped, {failed} failed.")
        self.refresh_samples()

    def _on_worker_failed(self, error: str) -> None:
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Pipeline failed.")
        self.append_log(f"Worker failed: {error}")
        QMessageBox.critical(self, "Pipeline Failed", error)

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
