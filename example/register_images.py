#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/06 18:16:11

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

# USER INPUTS
imreg.NCORES = 10                # Number of cores for registration 
videothresh = (100, 600) 
force_registration = False  # Force the registration of the data, even if the registration file already exists

# Select a folder
selected_folder = 'data/'

# Get all .mat files in the selected folder
mat_files = glob(os.path.join(selected_folder, '*.mat'))
mat_files = sorted(mat_files)

# If you want to process only a specific file, uncomment the next line and specify the file path
# mat_files = ['test_data2/nofibers_0CF_day7-01.mat']

# Register each mat file
failed_analyses = []
registering_times = []
for fname in mat_files:
    
    if 'warped' in fname:   # Skip already warped files
        continue

    # Dealing with paths
    sample = os.path.basename(fname).replace('.mat', '')
    path = os.path.dirname(fname)
    
    start = timer.time()  # Start timing the registration
    print(f"Registering {fname}...")

    # Load data and warp
    try:
        data = imu.load_data(fname, videothresh=videothresh)
        warped_data, displacements = imreg.register_all_frames(data)

        io.savemat(f'{path}/{sample}_warped.mat', {'warped_data': warped_data})
        imu.save_data(f'{path}/{sample}_warped.tif', warped_data)
        imu.save_data(f'{path}/{sample}.tif', data)
    except:
        failed_analyses.append(fname)
        print(f"Error registering {fname}. Skipping this file.")
        continue

    registering_times.append(timer.time() - start)

print(f'Registration completed for {len(mat_files)} files.')
print(f'Total registration time: {sum(registering_times):.2f} seconds.')
print(f'Average registration time per file: {np.mean(registering_times):.2f} seconds.')
print(f'Failed registrations: {len(failed_analyses)} files.')
if failed_analyses:
    print('Failed files:', failed_analyses)