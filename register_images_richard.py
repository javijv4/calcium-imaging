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
from scipy import io

import imregistration as imreg
import imutils as imu

import time as timer

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
imreg.NCORES = 10               # Number of cores for registration
videothresh = (0, 500)          # Richard stacks are ~400–500 frames
force_registration = False      # Force registration even if warped file already exists
save_tif = True
fix_cut = False

# Select a folder using the GUI
selected_folder = select_folder()

# Get all stack files in the selected folder
mat_files = list_stack_files(selected_folder)

# If you want to process only a specific file, uncomment the next line and specify the file path
mat_files = ['test_data_richard/TTN-mSca_GCaMP_488_50fps_40x_1002.tif']

# Register each file
failed_analyses = []
registering_times = []
for fname in mat_files:

    # Dealing with paths
    sample = os.path.splitext(os.path.basename(fname))[0]
    path = os.path.dirname(fname)

    if os.path.exists(f'{path}/{sample}_warped.mat') and not force_registration:
        print(f"File {fname} already registered. Skipping...")
        continue

    start = timer.time()  # Start timing the registration
    print(f"Registering {fname}...")

    # Load data and warp
    try:
        data = imu.load_data(fname, videothresh=videothresh, fix_cut=fix_cut)
        warped_data, displacements = imreg.register_all_frames(data)
        io.savemat(f'{path}/{sample}_warped.mat', {'warped_data': warped_data})
        if save_tif:
            imu.save_data(f'{path}/{sample}_warped.tif', warped_data)
            if not fname.lower().endswith(('.tif', '.tiff')):
                imu.save_data(f'{path}/{sample}.tif', data)
    except Exception as e:
        failed_analyses.append(fname)
        print(f"Error registering {fname}: {e}. Skipping this file.")
        continue

    registering_times.append(timer.time() - start)

print(f'Registration completed for {len(mat_files)} files.')
print(f'Total registration time: {sum(registering_times):.2f} seconds.')
print(f'Average registration time per file: {np.mean(registering_times) if registering_times else 0:.2f} seconds.')
print(f'Failed registrations: {len(failed_analyses)} files.')
if failed_analyses:
    print('Failed files:', failed_analyses)
