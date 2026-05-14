#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/07/07 13:54:13

@author: Javiera Jilberto Vallejos 
'''

import os
from tkinter import Tk, filedialog
from glob import glob

import numpy as np

from skimage import io as skio

import imregistration as imreg
import imutils as imu

import time


def select_folder(initialdir=None):
    """Open folder picker; starts in ``initialdir`` or the process current working directory."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    start = os.getcwd() if initialdir is None else initialdir
    folder_path = filedialog.askdirectory(
        title="Select a Folder",
        initialdir=start,
    )
    root.destroy()  # Close the Tkinter instance
    return folder_path


# USER INPUTS
framerate = 65.18
videothresh = (100, 600)            # To crop in time
pixelsize = 0.908
tissue_div_x = 6                    # For one region tissues
tissue_div_y = 4
imreg.NCORES = 10                   # Number of cores for registration 
threshold_value = 0                 # If 0, otsu thresholding is used
force_preprocessing = True          # Force the preprocessing of the data, even if the preprocessing file already exists
region_choice = 'grid'            # 'manual' or 'intensity' or 'grid' region selection

# Select a folder using the GUI
selected_folder = select_folder()

# Get all .mat files in the selected folder
mat_files = glob(os.path.join(selected_folder, '*.mat'))
mat_files = sorted(mat_files)

# Removing warped files
mat_files = [f for f in mat_files if not f.endswith('_warped.mat')]

# If you want to process only a specific file, uncomment the next line and specify the file path
# mat_files = ['test_data2/nofibers_iPSCCF_day7-01.mat']

preprocessing_times = []
for fname in mat_files:

    # Dealing with paths
    sample = os.path.basename(fname).replace('.mat', '')
    path = os.path.dirname(fname)

    if os.path.exists(f'{path}/{sample}_preprocessing.npz') and not force_preprocessing:
        print(f'Skipping {sample}, preprocessing already done.\n')
        continue

    start = time.time()  # Start timing the preprocessing
    print(f'Processing {sample}...')

    # Load data (T, H, W)
    data = imu.load_data(fname, videothresh=videothresh, fix_cut=False)

    # Get tissue mask
    print('Creating tissue mask...')
    mask = imu.get_tissue_mask(data)     # This will create a binary mask of the tissue
    skio.imsave(f'{path}/{sample}_tissue_mask.tif', mask.astype(np.uint8) * 255)  # Save mask for visualization

    # Divide the tissue in regions
    print("Selecting regions")
    data_2d, mask_2d = imu.stack_first_frame_for_rotate(data, mask)
    data_2d, mask_2d = imu.rotate_data(data_2d, mask_2d)
    
    if region_choice == 'manual':
        thresh, is_one_region = imu.divide_regions_choice(data_2d, mask_2d, nx=tissue_div_x, ny=tissue_div_y)
        if is_one_region:
            region_params = [tissue_div_x, tissue_div_y]
        else:
            region_params = [thresh, 0]
    elif region_choice == 'intensity':
        if data.ndim != 2:
            max_data = imu.max_over_time(data)
        else:
            max_data = data
        thresh = imu.find_tissue_regions_interactively(data_2d, mask_2d)
        is_one_region = False
        region_params = [thresh, 0]
    elif region_choice == 'grid':
        is_one_region = True
        region_params = [tissue_div_x, tissue_div_y]

    # Saving the data
    print('Saving preprocessing parameters..')
    np.savez(f'{path}/{sample}_preprocessing.npz', mask=mask.squeeze(), region_params=np.array(region_params), type=np.array([is_one_region]))

    # Save times
    preprocessing_times.append(time.time() - start)

    print('done.\n')
    
print(f'Preprocessing completed for {len(mat_files)} files.')
print(f'Total preprocessing time: {sum(preprocessing_times):.2f} seconds.')
print(f'Average preprocessing time per file: {np.mean(preprocessing_times):.2f} seconds.')
