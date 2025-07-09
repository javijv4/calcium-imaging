#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/07/07 13:54:13

@author: Javiera Jilberto Vallejos 
'''

import os
from tkinter import Tk, filedialog
from glob import glob

from matplotlib import pyplot as plt
import numpy as np

from scipy import io
from skimage import io as skio

import imregistration as imreg
import imutils as imu
import calcium_analysis as ca
import plotutils as pu


def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select a Folder")
    root.destroy()  # Close the Tkinter instance
    return folder_path


# USER INPUTS
framerate = 65.18
videothresh = (100, 600)            # To crop in time
pixelsize = 0.908
tissue_div_x = 2                    # For one region tissues
tissue_div_y = 6
imreg.NCORES = 10                   # Number of cores for registration 
threshold_value = 0                 # If 0, otsu thresholding is used
force_mask_creation = False         # Force the creation of a new tissue mask
force_registration = False          # Force the registration of images, this should always be set to False if you registered the images before

# Select a folder using the GUI
selected_folder = select_folder()

# Get all .mat files in the selected folder
mat_files = glob(os.path.join(selected_folder, '*.mat'))
mat_files = sorted(mat_files)

# Removing warped files
mat_files = [f for f in mat_files if not f.endswith('_warped.mat')]

for fname in mat_files:
    if ('nofibers' in fname) or ('0CF' in fname):
        is_one_region = False           # If several regions are to be analyzed, set to False.
    else:
        is_one_region = True            # If several regions are to be analyzed, set to False.


    # Dealing with paths
    sample = os.path.basename(fname).replace('.mat', '')
    path = os.path.dirname(fname)

    # Load data
    data = imu.load_data(fname, videothresh=videothresh)

    # Get tissue mask
    if force_mask_creation or not os.path.exists(f'{path}/{sample}_tissue_mask.tif'):
        print('Creating tissue mask...')
        mask = imu.get_tissue_mask(data)  # This will create a binary mask of the tissue
        skio.imsave(f'{path}/{sample}_tissue_mask.tif', mask.astype(np.uint8) * 255)  # Save mask for visualization
    else:
        print('Loading existing tissue mask...')
        # Load the saved mask
        mask = skio.imread(f'{path}/{sample}_tissue_mask.tif') // 255  # Load the mask and convert to binary (0s and 1s)
        mask = mask.astype(bool)  # Ensure it's binary for consistency

    # Register
    if is_one_region:
        # Register all frames
        if force_registration or not os.path.exists(f'{path}/{sample}_warped.mat'):
            print('Registering images...')
            warped_data, displacements = imreg.register_all_frames(data)

            # Save warped data
            io.savemat(f'{path}/{sample}_warped.mat', {'warped_data': warped_data})
        else:
            print('Loading warped images...')
            warped_data = io.loadmat(f'{path}/{sample}_warped.mat')['warped_data']

    else:
        warped_data = data

    # Rotate the data such that the tissue is vertical
    print('Rotating data...')
    warped_data, mask = imu.rotate_data(warped_data, mask)
    skio.imsave(f'{path}/{sample}_tissue_mask_rotated.tif', mask.astype(np.uint8) * 255)

    # Divide the tissue in regions
    if is_one_region:
        regions = imu.divide_tissue_in_regions(mask, ny=tissue_div_y, nx=tissue_div_x)
    else:
        regions = imu.find_tissue_regions(warped_data, mask)