"""GUI-facing application state models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppState:
    folder: Path | None = None
    selected_sample: str | None = None
    selected_path: Path | None = None

