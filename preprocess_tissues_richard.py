#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/08/16

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

STACK_EXTS = ('.mat', '.tif', '.tiff', '.nd2', '.czi')


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


def list_stack_files(folder):
    """Collect image stacks, preferring .tif over .nd2/.czi when both exist for the same stem."""
    by_stem = {}
    preference = {'.tif': 0, '.tiff': 0, '.mat': 1, '.nd2': 2, '.czi': 2}
    for ext in STACK_EXTS:
        for fname in glob(os.path.join(folder, f'*{ext}')):
            base = os.path.basename(fname)
            stem, file_ext = os.path.splitext(base)
            low = stem.lower()
            if 'warped' in low or low.endswith('_tissue_mask') or low.endswith('_tissue_mask_rotated'):
                continue
            prev = by_stem.get(stem)
            if prev is None or preference[file_ext.lower()] < preference[os.path.splitext(prev)[1].lower()]:
                by_stem[stem] = fname
    return sorted(by_stem.values())


# USER INPUTS
framerate = 50                  # 50fps stacks; Dual008 is 20 fps — change per folder if needed
videothresh = (0, 500)          # To crop in time
pixelsize = 0.283               # ~0.283 um (2x2 bin); Dual008 ~0.141 um (1x1)
tissue_div_x = 6                # For one region tissues
tissue_div_y = 4
imreg.NCORES = 10               # Number of cores for registration
threshold_value = 0             # If 0, otsu thresholding is used
force_preprocessing = True      # Force preprocessing even if preprocessing file already exists
region_choice = 'grid'          # 'manual' or 'intensity' or 'grid' region selection
fix_cut = False

# Select a folder using the GUI
selected_folder = select_folder()

# Get all stack files in the selected folder
mat_files = list_stack_files(selected_folder)

# If you want to process only a specific file, uncomment the next line and specify the file path
# mat_files = ['test_data_richard/TTN-mSca_GCaMP_488_50fps_40x_1002.tif']

preprocessing_times = []
for fname in mat_files:

    # Dealing with paths
    sample = os.path.splitext(os.path.basename(fname))[0]
    path = os.path.dirname(fname)

    if os.path.exists(f'{path}/{sample}_preprocessing.npz') and not force_preprocessing:
        print(f'Skipping {sample}, preprocessing already done.\n')
        continue

    start = time.time()  # Start timing the preprocessing
    print(f'Processing {sample}...')

    # Load data (T, H, W)
    data = imu.load_data(fname, videothresh=videothresh, fix_cut=fix_cut)

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
print(f'Average preprocessing time per file: {np.mean(preprocessing_times) if preprocessing_times else 0:.2f} seconds.')
