#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Convert a calcium image stack (.mat, .tif, .nd2, .czi) to a multi-page .tif."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.io import imsave

import imio


def load_stack(input_path, *, scene=0, channel=0, z=0):
    """Load a (T, H, W) stack from any supported format.

    ``.mat`` files use the ``data`` variable; if that is missing, ``warped_data``
    is used (registration output).
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext == ".mat":
        try:
            return imio.load_mat(input_path)
        except (KeyError, ValueError, OSError, TypeError, IndexError):
            mat = loadmat(input_path)
            if "warped_data" in mat:
                return mat["warped_data"]
            raise
    return imio.load_image(input_path, scene=scene, channel=channel, z=z)


def stack_to_tif(input_path, output_path=None, *, scene=0, channel=0, z=0):
    """Write ``input_path`` as a float32 multi-page TIFF."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".tif")
    else:
        output_path = Path(output_path)

    image_data = load_stack(input_path, scene=scene, channel=channel, z=z)
    imsave(output_path, np.asarray(image_data).astype(np.float32))
    print(f"Saved .tif file to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image stack (.mat, .tif/.tiff, .nd2, .czi) to .tif."
    )
    parser.add_argument("input", help="Path to the input stack")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .tif path (default: same name as input, .tif suffix)",
    )
    parser.add_argument("--scene", default=0, help="BioIO scene index or id (.nd2/.czi)")
    parser.add_argument("--channel", type=int, default=0, help="Channel index (.nd2/.czi)")
    parser.add_argument("--z", type=int, default=0, help="Z-plane index (.nd2/.czi)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file {input_path} does not exist.")

    scene = int(args.scene) if str(args.scene).isdigit() else args.scene
    stack_to_tif(
        input_path,
        args.output,
        scene=scene,
        channel=args.channel,
        z=args.z,
    )


if __name__ == "__main__":
    main()
