# -*- coding: utf-8 -*-
"""Pytest configuration: headless Matplotlib and golden-file regeneration flag."""

import os

import pytest


def pytest_configure(config):
    os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help=(
            "Only run test_regenerate_all_golden_files and write tests/expected/*.npz "
            "(registration, preprocess, analysis_inputs, analysis golden)."
        ),
    )


@pytest.fixture
def regen_golden(request):
    return request.config.getoption("--regen-golden")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--regen-golden"):
        regen = [i for i in items if "regenerate_all_golden" in i.nodeid]
        if regen:
            items[:] = regen
        return
    items[:] = [i for i in items if "regenerate_all_golden" not in i.nodeid]
