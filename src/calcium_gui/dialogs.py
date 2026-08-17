"""Dialogs for editing pipeline settings."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calcium_pipeline import PipelineSettings


class SettingsDialog(QDialog):
    """Edit pipeline settings from a top-bar menu."""

    def __init__(self, settings: PipelineSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        self._settings = PipelineSettings(**settings.__dict__)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self._build_general_tab()
        self._build_registration_tab()
        self._build_preprocess_tab()
        self._build_analysis_tab()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_settings()
        self.region_mode_combo.currentTextChanged.connect(self._update_threshold_state)
        self.use_videothresh_checkbox.toggled.connect(self._update_videothresh_state)
        self.use_threshold_checkbox.toggled.connect(self._update_threshold_state)
        self._update_videothresh_state()
        self._update_threshold_state()

    def _build_general_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self.framerate_spin = QDoubleSpinBox(tab)
        self.framerate_spin.setRange(0.01, 1000.0)
        self.framerate_spin.setDecimals(4)

        self.fix_cut_checkbox = QCheckBox("Enable weird-cut correction", tab)
        self.use_videothresh_checkbox = QCheckBox("Use manual time range", tab)
        self.videothresh_start_spin = QSpinBox(tab)
        self.videothresh_end_spin = QSpinBox(tab)
        for spin in (self.videothresh_start_spin, self.videothresh_end_spin):
            spin.setRange(0, 100000)

        video_row = QWidget(tab)
        video_layout = QHBoxLayout(video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(QLabel("Start", video_row))
        video_layout.addWidget(self.videothresh_start_spin)
        video_layout.addWidget(QLabel("End", video_row))
        video_layout.addWidget(self.videothresh_end_spin)

        self.scene_spin = QSpinBox(tab)
        self.channel_spin = QSpinBox(tab)
        self.z_spin = QSpinBox(tab)
        for spin in (self.scene_spin, self.channel_spin, self.z_spin):
            spin.setRange(0, 1000)

        form.addRow("Framerate", self.framerate_spin)
        form.addRow("", self.fix_cut_checkbox)
        form.addRow("", self.use_videothresh_checkbox)
        form.addRow("Time range", video_row)
        form.addRow("Scene", self.scene_spin)
        form.addRow("Channel", self.channel_spin)
        form.addRow("Z index", self.z_spin)
        self.tabs.addTab(tab, "General")

    def _build_registration_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self.registration_cores_spin = QSpinBox(tab)
        self.registration_cores_spin.setRange(1, 128)
        self.force_registration_checkbox = QCheckBox("Overwrite existing warped output", tab)

        form.addRow("CPU cores", self.registration_cores_spin)
        form.addRow("", self.force_registration_checkbox)
        self.tabs.addTab(tab, "Registration")

    def _build_preprocess_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self.force_preprocessing_checkbox = QCheckBox("Overwrite existing preprocessing output", tab)
        self.tissue_div_x_spin = QSpinBox(tab)
        self.tissue_div_y_spin = QSpinBox(tab)
        for spin in (self.tissue_div_x_spin, self.tissue_div_y_spin):
            spin.setRange(1, 128)

        self.region_mode_combo = QComboBox(tab)
        self.region_mode_combo.addItems(["grid", "threshold"])

        self.use_threshold_checkbox = QCheckBox("Use explicit threshold value", tab)
        self.threshold_spin = QDoubleSpinBox(tab)
        self.threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.threshold_spin.setDecimals(4)

        form.addRow("", self.force_preprocessing_checkbox)
        form.addRow("Grid X", self.tissue_div_x_spin)
        form.addRow("Grid Y", self.tissue_div_y_spin)
        form.addRow("Region mode", self.region_mode_combo)
        form.addRow("", self.use_threshold_checkbox)
        form.addRow("Threshold", self.threshold_spin)
        self.tabs.addTab(tab, "Preprocess")

    def _build_analysis_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self.analysis_cores_spin = QSpinBox(tab)
        self.analysis_cores_spin.setRange(1, 128)

        self.width_factor_spin = QDoubleSpinBox(tab)
        self.width_factor_spin.setRange(0.001, 1.0)
        self.width_factor_spin.setDecimals(4)
        self.plot_all_traces_checkbox = QCheckBox("Save individual trace figure", tab)

        form.addRow("CPU cores", self.analysis_cores_spin)
        form.addRow("Peak width factor", self.width_factor_spin)
        form.addRow("", self.plot_all_traces_checkbox)
        self.tabs.addTab(tab, "Analysis")

    def _load_settings(self) -> None:
        settings = self._settings
        self.framerate_spin.setValue(settings.framerate)
        self.fix_cut_checkbox.setChecked(settings.fix_cut)
        self.use_videothresh_checkbox.setChecked(settings.videothresh() is not None)
        self.videothresh_start_spin.setValue(settings.videothresh_start or 0)
        self.videothresh_end_spin.setValue(settings.videothresh_end or 0)
        self.scene_spin.setValue(settings.scene)
        self.channel_spin.setValue(settings.channel)
        self.z_spin.setValue(settings.z_index)
        self.registration_cores_spin.setValue(settings.registration_cores)
        self.force_registration_checkbox.setChecked(settings.force_registration)
        self.force_preprocessing_checkbox.setChecked(settings.force_preprocessing)
        self.tissue_div_x_spin.setValue(settings.tissue_div_x)
        self.tissue_div_y_spin.setValue(settings.tissue_div_y)
        self.region_mode_combo.setCurrentText(settings.region_mode)
        self.use_threshold_checkbox.setChecked(settings.threshold_value is not None)
        self.threshold_spin.setValue(settings.threshold_value or 0.0)
        self.analysis_cores_spin.setValue(settings.analysis_cores)
        self.width_factor_spin.setValue(settings.width_factor)
        self.plot_all_traces_checkbox.setChecked(settings.plot_all_traces)

    def _update_videothresh_state(self) -> None:
        enabled = self.use_videothresh_checkbox.isChecked()
        self.videothresh_start_spin.setEnabled(enabled)
        self.videothresh_end_spin.setEnabled(enabled)

    def _update_threshold_state(self) -> None:
        enabled = self.region_mode_combo.currentText() == "threshold"
        self.use_threshold_checkbox.setEnabled(enabled)
        use_explicit = enabled and self.use_threshold_checkbox.isChecked()
        self.threshold_spin.setEnabled(use_explicit)

    def settings(self) -> PipelineSettings:
        videothresh_start = self.videothresh_start_spin.value() if self.use_videothresh_checkbox.isChecked() else None
        videothresh_end = self.videothresh_end_spin.value() if self.use_videothresh_checkbox.isChecked() else None
        threshold_value = self.threshold_spin.value() if (
            self.region_mode_combo.currentText() == "threshold" and self.use_threshold_checkbox.isChecked()
        ) else None
        return PipelineSettings(
            framerate=self.framerate_spin.value(),
            videothresh_start=videothresh_start,
            videothresh_end=videothresh_end,
            fix_cut=self.fix_cut_checkbox.isChecked(),
            scene=self.scene_spin.value(),
            channel=self.channel_spin.value(),
            z_index=self.z_spin.value(),
            registration_cores=self.registration_cores_spin.value(),
            analysis_cores=self.analysis_cores_spin.value(),
            tissue_div_x=self.tissue_div_x_spin.value(),
            tissue_div_y=self.tissue_div_y_spin.value(),
            force_registration=self.force_registration_checkbox.isChecked(),
            force_preprocessing=self.force_preprocessing_checkbox.isChecked(),
            region_mode=self.region_mode_combo.currentText(),
            threshold_value=threshold_value,
            width_factor=self.width_factor_spin.value(),
            plot_all_traces=self.plot_all_traces_checkbox.isChecked(),
        )

