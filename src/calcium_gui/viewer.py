"""File viewers for images, stacks, CSVs, and parameter outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calcium_pipeline import load_stack_for_viewer


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _array_to_qimage(array: np.ndarray) -> QImage:
    array = np.asarray(array)
    if array.ndim == 2:
        frame = np.ascontiguousarray(_normalize_to_uint8(array))
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_Grayscale8).copy()
    if array.ndim == 3 and array.shape[2] == 3:
        frame = np.ascontiguousarray(_normalize_to_uint8(array))
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888).copy()
    if array.ndim == 3 and array.shape[2] == 4:
        frame = np.ascontiguousarray(_normalize_to_uint8(array))
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGBA8888).copy()
    raise ValueError(f"Unsupported array shape for display: {array.shape}")


class ImageLabel(QLabel):
    """A QLabel that rescales the original pixmap on resize."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self._pixmap = QPixmap()

    def set_viewer_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap.isNull():
            self.clear()
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class FileViewer(QWidget):
    """Main viewer used on the right side of the GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack_data: np.ndarray | None = None

        layout = QVBoxLayout(self)
        self.path_label = QLabel("Select a file from the sample tree to preview it.", self)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.pages = QStackedWidget(self)
        layout.addWidget(self.pages, 1)

        self.placeholder = QLabel("No file selected", self)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.pages.addWidget(self.placeholder)

        self.image_label = ImageLabel(self)
        self.image_scroll = QScrollArea(self)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        self.pages.addWidget(self.image_scroll)

        self.stack_page = QWidget(self)
        stack_layout = QVBoxLayout(self.stack_page)
        self.stack_image_label = ImageLabel(self.stack_page)
        self.stack_scroll = QScrollArea(self.stack_page)
        self.stack_scroll.setWidgetResizable(True)
        self.stack_scroll.setWidget(self.stack_image_label)
        stack_layout.addWidget(self.stack_scroll, 1)
        self.frame_label = QLabel("Frame 1 / 1", self.stack_page)
        stack_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Horizontal, self.stack_page)
        self.frame_slider.valueChanged.connect(self._update_stack_frame)
        stack_layout.addWidget(self.frame_slider)
        self.pages.addWidget(self.stack_page)

        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pages.addWidget(self.table)

        self.text_view = QPlainTextEdit(self)
        self.text_view.setReadOnly(True)
        self.pages.addWidget(self.text_view)

        self.pages.setCurrentWidget(self.placeholder)

    def clear(self) -> None:
        self._stack_data = None
        self.path_label.setText("Select a file from the sample tree to preview it.")
        self.pages.setCurrentWidget(self.placeholder)

    def show_path(self, path: str | Path) -> None:
        source = Path(path)
        self.path_label.setText(str(source))
        suffix = source.suffix.lower()
        if suffix == ".csv":
            self._show_csv(source)
            return
        if suffix == ".npz":
            self._show_npz(source)
            return
        if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            self._show_pixmap(QPixmap(str(source)))
            return
        if suffix in {".mat", ".tif", ".tiff", ".nd2", ".czi"}:
            self._show_stack(source)
            return
        self.text_view.setPlainText(f"Preview is not implemented for {source.name}.")
        self.pages.setCurrentWidget(self.text_view)

    def _show_pixmap(self, pixmap: QPixmap) -> None:
        self.image_label.set_viewer_pixmap(pixmap)
        self.pages.setCurrentWidget(self.image_scroll)

    def _show_stack(self, source: Path) -> None:
        stack = load_stack_for_viewer(source)
        self._stack_data = stack
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(stack.shape[0] - 1, 0))
        self.frame_slider.setEnabled(stack.shape[0] > 1)
        self.frame_slider.setValue(0)
        self._update_stack_frame(0)
        self.pages.setCurrentWidget(self.stack_page)

    def _update_stack_frame(self, value: int) -> None:
        if self._stack_data is None:
            return
        frame = self._stack_data[value]
        pixmap = QPixmap.fromImage(_array_to_qimage(frame))
        self.stack_image_label.set_viewer_pixmap(pixmap)
        self.frame_label.setText(f"Frame {value + 1} / {self._stack_data.shape[0]}")

    def _show_csv(self, source: Path) -> None:
        with source.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        else:
            self.table.setColumnCount(len(rows[0]))
            self.table.setHorizontalHeaderLabels(rows[0])
            self.table.setRowCount(max(len(rows) - 1, 0))
            for row_index, row in enumerate(rows[1:]):
                for col_index, value in enumerate(row):
                    self.table.setItem(row_index, col_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.pages.setCurrentWidget(self.table)

    def _show_npz(self, source: Path) -> None:
        data = np.load(source)
        lines = [f"{key}: shape={np.asarray(data[key]).shape}, dtype={np.asarray(data[key]).dtype}" for key in data.files]
        self.text_view.setPlainText("\n".join(lines) if lines else "(empty npz)")
        self.pages.setCurrentWidget(self.text_view)

