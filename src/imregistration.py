#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/10 17:59:26

@author: Javiera Jilberto Vallejos 
'''

import numpy as np

from skimage import img_as_float32, filters

import itk
from tqdm import tqdm

from imutils import normalize_image

from time import time
import pathlib

filepath = pathlib.Path(__file__).parent.resolve()

NCORES = 4


def compute_displacements(data):
    """
    Register each frame to the first frame; return displacement field stack.

    data : (T, H, W) — smoothed internally for registration.
    Returns displacements (T, H, W, 2); frame 0 is zero.
    """
    nframes = data.shape[0]
    work = filters.gaussian(data, sigma=1)

    displacements = np.zeros((nframes, data.shape[1], data.shape[2], 2))
    moving = work[0, :, :]
    for t in tqdm(range(1, nframes)):
        fixed = work[t, :, :]
        _, displacement_field = register_images(fixed, moving)
        displacements[t, :, :, :] = displacement_field

    return displacements


def warp_stack(data_og, displacements):
    """Apply per-frame displacements to align frames to frame 0."""
    tlen, h, w = data_og.shape
    i = np.arange(h)
    j = np.arange(w)
    I, J = np.meshgrid(i, j)
    IJ = np.vstack([I.ravel(), J.ravel()]).T

    warped_data = np.zeros_like(data_og)

    print("Warping data")
    for t in tqdm(range(tlen)):
        disp = np.vstack(
            [
                displacements[t, :, :, 0].ravel(),
                displacements[t, :, :, 1].ravel(),
            ]
        ).T
        ij = IJ - disp.astype(int)
        ij = np.clip(ij, 0, h - 1)

        warped_data[t, :, :] = data_og[t, ij[:, 1], ij[:, 0]].reshape(h, w)

    vmin = data_og.min()
    vmax = data_og.max()
    warped_data = normalize_image(warped_data) * (vmax - vmin) + vmin

    return warped_data


def register_all_frames(data):
    """Register all frames to the first frame; return warped (T,H,W) and displacements."""
    start = time()
    data_og = data.copy()
    displacements = compute_displacements(data_og)
    warped_data = warp_stack(data_og, displacements)
    print("Registration took: ", time() - start, " seconds")
    return warped_data, displacements


def get_displacement_field(movingArray, resultParameters):
    movingArray = img_as_float32(np.ascontiguousarray(movingArray))
    movingImage = itk.GetImageFromArray(movingArray)
    deformation_field = itk.transformix_deformation_field(movingImage, resultParameters)
    defArray = itk.GetArrayFromImage(deformation_field).astype(float)
    return defArray


def register_images(fixed, moving, init_transform_params=None):
    fixed_array = np.ascontiguousarray(fixed.astype(np.float32))
    moving_array = np.ascontiguousarray(moving.astype(np.float32))

    fixed_image = itk.GetImageFromArray(fixed_array)
    moving_image = itk.GetImageFromArray(moving_array)

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(f'{filepath}/parameters_BSpline.txt')

    warped_img, transform_params = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        number_of_threads=NCORES,
        log_to_console=False,
        initial_transform_parameter_object=init_transform_params,
    )

    warped_img = itk.GetArrayFromImage(warped_img)
    warped_img[warped_img < moving.min()] = moving.min()

    displacement_field = get_displacement_field(moving, transform_params)

    if init_transform_params is None:
        return warped_img, displacement_field
    else:
        return warped_img, displacement_field, transform_params
