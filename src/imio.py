#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Image-format IO: MATLAB, TIFF, and BioIO stacks (.nd2, .czi)."""

from pathlib import Path

import numpy as np
import tifffile
from scipy.io import loadmat

from bioio import BioImage


def _is_mat73(path):
    with open(path, 'rb') as f:
        header = f.read(128)
    return b'HDF5' in header


def load_mat(fname):
    """Load MATLAB variable ``data`` as a (T, H, W) float array."""
    import h5py

    if _is_mat73(fname):
        with h5py.File(fname, 'r') as f:
            data_ref = f['data'][()][0, 0]
            data_group = f[data_ref]
            images = []
            for i in range(len(data_group)):
                ref = data_group[str(i)][0]
                image = f[ref][()]
                images.append(image)
    else:
        mat = loadmat(fname, variable_names=['data'])
        data_struct = mat['data'][0][0]
        images = []
        for i in range(len(data_struct)):
            images.append(data_struct[i][0])

    return np.stack(images, axis=0)


def load_tif(fname):
    """Load a multi-page TIFF as (T, H, W)."""
    arr = tifffile.imread(fname)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF array, got shape {arr.shape}")
    return np.asarray(arr)


def open_image(fname):
    """Return a BioIO ``BioImage`` without loading pixel data."""
    return BioImage(fname)


def _set_scene(img, scene):
    scenes = img.scenes
    if isinstance(scene, int):
        if scene < 0 or scene >= len(scenes):
            raise ValueError(
                f"scene={scene} out of range; {len(scenes)} scene(s): {list(scenes)}"
            )
        img.set_scene(scene)
        return
    try:
        img.set_scene(scene)
    except Exception as exc:
        raise ValueError(
            f"scene={scene!r} is not valid; available scenes: {list(scenes)}"
        ) from exc


def _check_index(name, value, size, extra=""):
    if value < 0 or value >= size:
        raise ValueError(f"{name}={value} out of range; {name[0].upper()}={size}{extra}")


def load_stack(fname, *, scene=0, channel=0, z=0):
    """Load a BioIO-supported stack as (T, H, W).

    Defaults: first scene, channel 0, Z plane 0. Missing time is a singleton
    frame ``(1, H, W)``.
    """
    img = BioImage(fname)
    _set_scene(img, scene)

    dims = img.dims
    channel_names = getattr(img, "channel_names", None)
    extra_c = f" (channels: {list(channel_names)})" if channel_names else ""
    _check_index("channel", channel, dims.C, extra_c)
    _check_index("z", z, dims.Z)

    arr = img.get_image_data("TYX", C=channel, Z=z)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array from BioIO, got shape {arr.shape}")
    return np.asarray(arr)


def load_image(fname, *, scene=0, channel=0, z=0):
    """Load ``fname`` as (T, H, W) by extension (``.mat``, ``.tif``, ``.nd2``, ``.czi``)."""
    ext = Path(fname).suffix.lower()
    if ext == ".mat":
        return load_mat(fname)
    if ext in (".tif", ".tiff"):
        return load_tif(fname)
    if ext in (".nd2", ".czi"):
        return load_stack(fname, scene=scene, channel=channel, z=z)
    raise ValueError(
        f"Unsupported file type: {ext} (use .mat, .tif/.tiff, .nd2, or .czi)"
    )
