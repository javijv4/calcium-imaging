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

filepath=pathlib.Path(__file__).parent.resolve()

NCORES=4

def register_all_frames(data):
    start = time()
    
    nframes = data.shape[2]
    data_og = data.copy()

    # Filter data
    data = filters.gaussian(data, sigma=1)

    # Register all frames
    displacements = np.zeros((data.shape[0], data.shape[1], nframes, 2))
    moving = data[:, :, 0]
    for i in tqdm(range(1, nframes)):
        fixed = data[:, :, i]

        _, displacement_field = register_images(fixed, moving)

        # Update data
        displacements[:, :, i, :] = displacement_field

    # Warp data
    # Generate grid
    i = np.arange(data.shape[0])
    j = np.arange(data.shape[1])
    I, J = np.meshgrid(i, j)

    IJ = np.vstack([I.ravel(), J.ravel()]).T

    # Grab intensities
    warped_data = np.zeros_like(data)

    print("Warping data")
    for i in tqdm(range(data.shape[2])):
        disp = np.vstack([displacements[:,:,i,0].ravel(), displacements[:,:,i,1].ravel()]).T
        ij = IJ - disp.astype(int)
        ij = np.clip(ij, 0, data.shape[0]-1)

        warped_data[:,:,i] = data_og[ij[:,1], ij[:,0], i].reshape(data.shape[0], data.shape[1])


    # Rescale data to match original
    vmin = data_og.min()
    vmax = data_og.max()
    warped_data = normalize_image(warped_data) * (vmax - vmin) + vmin

    print("Registration took: ", time() - start, " seconds")
    
    return warped_data, displacements


def get_displacement_field(movingArray, resultParameters):
    movingArray = img_as_float32(movingArray)
    movingImage = itk.GetImageFromArray(movingArray)
    deformation_field = itk.transformix_deformation_field(movingImage, resultParameters)
    defArray = itk.GetArrayFromImage(deformation_field).astype(float)
    return defArray


def register_images(fixed, moving, init_transform_params=None):
    fixed_array = fixed.astype(np.float32)
    moving_array = moving.astype(np.float32)

    fixed_image = itk.GetImageFromArray(fixed_array)
    moving_image = itk.GetImageFromArray(moving_array)

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()

    # Load custom parameter maps from .txt file
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(f'{filepath}/parameters_BSpline.txt')

    # Load Elastix Image Filter Object
    warped_img, transform_params = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object,
    number_of_threads=NCORES,
    log_to_console=False, initial_transform_parameter_object=init_transform_params)

    warped_img = np.array(warped_img)
    warped_img[warped_img < moving.min()] = moving.min()

    displacement_field = get_displacement_field(moving, transform_params)

    if init_transform_params is None:
        return warped_img, displacement_field
    else:
        return warped_img, displacement_field, transform_params

