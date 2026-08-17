import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="PySide6 not installed")
def test_main_window_builds():
    from PySide6.QtWidgets import QApplication

    from calcium_gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "Calcium Imaging GUI"
    assert window.tree.headerItem().text(0) == "Samples"
