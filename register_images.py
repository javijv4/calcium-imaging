#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/06 18:16:11

@author: Javiera Jilberto Vallejos 
'''

import os
from tkinter import Tk, filedialog
from glob import glob

from scipy import io

import imregistration as imreg
import imutils as imu

def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select a Folder")
    return folder_path

if __name__ == "__main__":

    # USER INPUTS
    imreg.NCORES = 10                # Number of cores for registration 
    videothresh = (200, 500)         # To crop in time

    # Select a folder using the GUI
    selected_folder = select_folder()

    # Get all .mat files in the selected folder
    mat_files = glob(os.path.join(selected_folder, '*.mat'))
    mat_files = sorted(mat_files)
    
    # Register each mat file
    for fname in mat_files:
        if ('nofibers' in fname) or ('0CF' in fname):
            continue           # These files do not need to be registered. @Maggie: Is this correct?
        
        print(f"Registering {fname}...")

        # Dealing with paths
        sample = os.path.basename(fname).replace('.mat', '')
        path = os.path.dirname(fname)

        # Load data and warp
        try:
            data = imu.load_data(fname, videothresh=videothresh)
            warped_data, displacements = imreg.register_all_frames(data)

            io.savemat(f'{path}/{sample}_warped.mat', {'warped_data': warped_data})
        except:
            print(f"Error registering {fname}. Skipping this file.")
            continue
