#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/16 18:16:11

@author: Javiera Jilberto Vallejos 
'''

import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import io
from skimage import io as skio
from skimage import filters

import imregistration as imreg
import imutils as imu
import calcium_analysis as ca
import plotutils as pu

# USER INPUTS
framerate = 65.18
videothresh = (200, 500)         # To crop in time
pixelsize = 0.908
tissue_div_x = 2                # For one region tissues
tissue_div_y = 6
imreg.NCORES = 10                # Number of cores for registration 
threshold_value = 0              # If 0, otsu thresholding is used

force_registration = True
force_mask_creation = False     # Force the creation of a new tissue mask

fname = 'test_data2/fibers_iPSCCF_day7-06.mat'

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
    mask = imu.get_tissue_mask(data, is_one_region=is_one_region)  # This will create a binary mask of the tissue
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

# Divide the tissue in regions
if is_one_region:
    regions = imu.divide_tissue_in_regions(mask, ny=tissue_div_y, nx=tissue_div_x)
else:
    regions = imu.find_tissue_regions(data, mask)

# Evaluate intensities in the whole tissue
if is_one_region:
    tissue_trace = imu.evaluate_regional_intensities(warped_data, mask.astype(int))[:,0]

    filtered_traces, max_peaks_idx, min_peaks_idx = ca.analyze_trace(tissue_trace)
    if len(max_peaks_idx) <= 2:     # No peaks were found
        bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = 0, 0, 0, 0, 0
    else:
        bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(filtered_traces, max_peaks_idx, 
                                                                            min_peaks_idx, framerate)
    tissue_calcium_trace = ca.CalciumTrace(filtered_traces, max_peaks_idx, min_peaks_idx, 0, 
                                bpm, bpm_std, timing_irregularity, upstroke_time, amplitude)


# Evaluate intensities in the regions
traces = imu.evaluate_regional_intensities(warped_data, regions)

# Analyze traces
calcium_traces = []
valid_regions = []
for i, trace in enumerate(traces.T):
    filtered_traces, max_peaks_idx, min_peaks_idx = ca.analyze_trace(trace)
    if len(max_peaks_idx) <= 2:     # No peaks were found
        continue
    bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(filtered_traces, max_peaks_idx, 
                                                                          min_peaks_idx, framerate)
    ctrace = ca.CalciumTrace(filtered_traces, max_peaks_idx, min_peaks_idx, i+1, 
                             bpm, bpm_std, timing_irregularity, upstroke_time, amplitude)
    calcium_traces.append(ctrace)
    valid_regions.append(ctrace.region)  # Add region number to the list of regions


# Delete non-valid regions
valid_mask = np.isin(regions, valid_regions)  # Create a mask for valid regions
regions[~valid_mask] = 0  # Set non-valid regions to 0
aux_regions = np.copy(regions)  # Create a copy of the regions for later use
for i, ctrace in enumerate(calcium_traces):
    regions[aux_regions == ctrace.region] = i+1  # Update regions in the mask
    ctrace.region = i+1  # Update region numbers to be consecutive
    


# Synchronicity
traces = np.vstack([[ctrace.trace for ctrace in calcium_traces]]).T
synchronicity = np.mean(np.corrcoef(traces.T))


# Tissue outputs
if is_one_region:
    header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'upstroke time', 'amplitude']
    np.savetxt(f'{path}/{sample}_output.csv', 
            np.array([[sample, tissue_calcium_trace.bpm, tissue_calcium_trace.bpm_std, 
                        tissue_calcium_trace.timing_irregularity, synchronicity,
                        tissue_calcium_trace.upstroke_time, tissue_calcium_trace.amplitude]]), 
            delimiter=',', fmt='%s', header=','.join(header), comments='')
else:
    header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'upstroke time', 'amplitude']
    np.savetxt(f'{path}/{sample}_output.csv', 
            np.array([[sample, '', '', '', 
                        synchronicity, '', '']]), 
            delimiter=',', fmt='%s', header=','.join(header), comments='')
    

# Region outputs
outputs = []
for i, ctrace in enumerate(calcium_traces):
    header = ['Sample Name', 'Region', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'upstroke time', 'amplitude']
    outputs.append([sample, ctrace.region, ctrace.bpm, ctrace.bpm_std,
                    ctrace.timing_irregularity, synchronicity, ctrace.upstroke_time, ctrace.amplitude])
    
np.savetxt(f'{path}/{sample}_region_output.csv',
            np.array(outputs), 
            delimiter=',', fmt='%s', header=','.join(header), comments='')


# Raw outputs
time = np.arange(traces.shape[0]) / framerate  # Time in seconds
traces_raw = [time]
traces_raw += [tissue_calcium_trace.trace] if is_one_region else [np.zeros(len(time))]
traces_raw += [trace.trace for trace in calcium_traces]
traces_raw = np.vstack(traces_raw).T

np.savetxt(f'{path}/{sample}_raw_output.csv',
            traces_raw, 
            delimiter=',', fmt='%s', header='Time,' + ','.join(['Tissue'] + [f'Region {i+1}' for i in range(len(calcium_traces))]), comments='')

# Plot
pu.plot_regions_traces(warped_data, regions, calcium_traces)
plt.savefig(f'{path}/{sample}_regions_traces.png', dpi=300, bbox_inches='tight')