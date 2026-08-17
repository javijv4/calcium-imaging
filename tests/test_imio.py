#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for imio format loaders."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

import imio


def test_load_tif_3d(tmp_path):
    path = tmp_path / "stack.tif"
    expected = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    tifffile.imwrite(path, expected)
    out = imio.load_tif(path)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, expected)


def test_load_tif_2d_becomes_singleton_time(tmp_path):
    path = tmp_path / "frame.tif"
    frame = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(path, frame)
    out = imio.load_tif(path)
    assert out.shape == (1, 3, 4)
    np.testing.assert_array_equal(out[0], frame)


def test_load_stack_defaults_channel0_z0():
    volume = np.arange(3 * 2 * 4 * 5 * 6, dtype=np.float32).reshape(3, 2, 4, 5, 6)
    mock_img = MagicMock()
    mock_img.scenes = ("Image:0",)
    mock_img.dims.C = 2
    mock_img.dims.Z = 4
    mock_img.channel_names = ["A", "B"]
    mock_img.get_image_data.return_value = volume[:, 0, 0, :, :]

    with patch("imio.BioImage", return_value=mock_img) as bio:
        out = imio.load_stack("fake.nd2")

    bio.assert_called_once_with("fake.nd2")
    mock_img.set_scene.assert_called_once_with(0)
    mock_img.get_image_data.assert_called_once_with("TYX", C=0, Z=0)
    assert out.shape == (3, 5, 6)
    np.testing.assert_array_equal(out, volume[:, 0, 0, :, :])


def test_load_stack_channel_and_z():
    tyx = np.ones((2, 4, 5), dtype=np.float32)
    mock_img = MagicMock()
    mock_img.scenes = ("Image:0", "Image:1")
    mock_img.dims.C = 3
    mock_img.dims.Z = 2
    mock_img.channel_names = ["A", "B", "C"]
    mock_img.get_image_data.return_value = tyx

    with patch("imio.BioImage", return_value=mock_img):
        out = imio.load_stack("fake.czi", scene=1, channel=2, z=1)

    mock_img.set_scene.assert_called_once_with(1)
    mock_img.get_image_data.assert_called_once_with("TYX", C=2, Z=1)
    assert out.shape == (2, 4, 5)


def test_load_stack_invalid_channel():
    mock_img = MagicMock()
    mock_img.scenes = ("Image:0",)
    mock_img.dims.C = 1
    mock_img.dims.Z = 1
    mock_img.channel_names = ["GCaMP"]

    with patch("imio.BioImage", return_value=mock_img):
        with pytest.raises(ValueError, match="channel=1"):
            imio.load_stack("fake.nd2", channel=1)


def test_load_image_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        imio.load_image("file.png")
