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
from skimage import img_as_int

import imregistration as imreg
import imutils as imu
import calcium_analysis as ca
import plotutils as pu

from tkinter import Tk, filedialog
import glob

import time as timer

# USER INPUTS
framerate = 65.18
videothresh = (100, 600)         # To crop in time
pixelsize = 0.908
tissue_div_x = 2                # For one region tissues
tissue_div_y = 6
imreg.NCORES = 8                # Number of cores for registration 
threshold_value = 0              # If 0, otsu thresholding is used
width_factor = 1/10
plot_all_traces = True
rotate_orientation = None

# Select a folder
selected_folder = 'data/'

# Get all .mat files in the selected folder
filenames = []
# filenames = ['test_data2/nofibers_0CF_day7-01.mat'] # If you want to process only a specific file, uncomment the next line and specify the file path
if len(filenames) == 0:
    mat_files = glob.glob(os.path.join(selected_folder, '*.mat'))
mat_files = sorted(mat_files)

# Removing warped files
filenames = [f for f in mat_files if not f.endswith('_warped.mat')]

processing_times = []
failed_analyses = []
all_outputs = []
for fname in filenames:
    try:
        start = timer.time()  # Start timing the preprocessing

        # Dealing with paths
        sample = os.path.basename(fname).replace('.mat', '')
        path = os.path.dirname(fname)

        print(f'Processing {sample}...')

        # Load preprocessing info
        if os.path.exists(f'{path}/{sample}_preprocessing.npz'):
            print("Loading preprocessing info...")
            preprocess_info = np.load(f'{path}/{sample}_preprocessing.npz')
        else:
            print("Preprocessing file not found. Please run the preprocessing script first.")
            continue

        is_one_region = preprocess_info['type'][0]
        region_params = preprocess_info['region_params']
        mask = preprocess_info['mask']  # Ensure mask is a 2D array

        # Register
        if os.path.exists(f'{path}/{sample}_warped.mat'):
            print('Loading warped images...')
            warped_data = io.loadmat(f'{path}/{sample}_warped.mat')['warped_data']
        else:
            print('Warped images not found. Please run the preprocessing script first.')
            # continue

        # Rotate the data such that the tissue is vertical
        if rotate_orientation is not None:
            print('Rotating data...')
            warped_data, mask = imu.rotate_data_cv2(warped_data, mask, target_orientation=rotate_orientation)
            first_frame = warped_data[0]
            skio.imsave(f'{path}/{sample}_tissue_mask_rotated.tif', mask.astype(np.uint8) * 255)
            skio.imsave(f'{path}/{sample}_warped_tissue_rotated.tif', warped_data.astype(np.int16), 
                        check_contrast=False)

        # Divide the tissue in regions
        if is_one_region:
            regions = imu.divide_tissue_in_regions(mask=mask, nx=region_params[0], ny=region_params[1])
        else:
            regions = imu.apply_threshold(final_threshold=region_params[0], data=first_frame, tissue_mask=mask)


        # Evaluate intensities in the whole tissue
        print("Analyzing tissue intensities...")
        if is_one_region:
            tissue_trace = imu.evaluate_regional_intensities(warped_data, mask.astype(int))[:,0]

            filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace(tissue_trace, min_peaks_found=1, detrend_trace=False)

            if len(max_peaks_idx) <= 2:     # No peaks were found
                bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = 0, 0, 0, 0, 0
            else:
                bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(filtered_trace, max_peaks_idx, 
                                                                                    min_peaks_idx, framerate)
            tissue_calcium_trace = ca.CalciumTrace(filtered_trace, max_peaks_idx, min_peaks_idx, 0, 
                                        bpm, bpm_std, timing_irregularity, upstroke_time, amplitude)

            
        # Evaluate intensities in the regions
        traces = imu.evaluate_regional_intensities(warped_data, regions)

        # Filter traces
        filtered_traces = []
        max_peaks = []
        min_peaks = []
        for i, trace in enumerate(traces.T):
            filtered_trace, max_peaks_idx, min_peaks_idx = ca.analyze_trace_fft(trace, framerate=framerate, width_factor=width_factor)
            filtered_traces.append(filtered_trace)  # Store the filtered trace for later use
            max_peaks.append(max_peaks_idx)
            min_peaks.append(min_peaks_idx)


        # Plot all traces
        if plot_all_traces:
            ntraces = len(filtered_traces)
            fig, axs = plt.subplots(ntraces, 1, figsize=(10, 2*ntraces), sharex=True)
            time_trace = np.arange(len(filtered_traces[0])) / framerate
            for i, trace in enumerate(filtered_traces):
                max_peaks_idx = max_peaks[i]
                min_peaks_idx = min_peaks[i]

                axs[i].plot(time_trace, trace, 'k', label='Filtered Trace')
                if len(max_peaks_idx) > 0:
                    axs[i].plot(max_peaks_idx / framerate, trace[max_peaks_idx], 'ro', label='Max Peaks')
                if len(min_peaks_idx) > 0:
                    axs[i].plot(min_peaks_idx / framerate, trace[min_peaks_idx], 'bo', label='Min Peaks')
                axs[i].set_ylabel('Intensity')

            plt.tight_layout()
            plt.savefig(f'{path}/{sample}_all_individual_traces.png', dpi=180, bbox_inches='tight')

        # Plot initial traces
        filtered_traces = np.array(filtered_traces)
        fig1, fig2 = pu.plot_regions_traces(first_frame, regions, filtered_traces, framerate=framerate)
        fig1.savefig(f'{path}/{sample}_all_regions.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{path}/{sample}_all_traces.png', dpi=300, bbox_inches='tight')

        # Synchronicity
        synchronicity = np.mean(np.corrcoef(filtered_traces))
        print(f'Synchronicity: {synchronicity:.6f}')

        # Analyze traces
        calcium_traces = []
        valid_regions = []
        for i, trace in enumerate(traces.T):
            filtered_trace = filtered_traces[i]
            max_peaks_idx = max_peaks[i]
            min_peaks_idx = min_peaks[i]
            max_peaks.append(max_peaks_idx)
            min_peaks.append(min_peaks_idx)

            if len(max_peaks_idx) <= 2:     # No peaks were found
                continue
            bpm, bpm_std, timing_irregularity, upstroke_time, amplitude = ca.trace_outputs(filtered_trace, max_peaks_idx, 
                                                                                min_peaks_idx, framerate)
            ctrace = ca.CalciumTrace(filtered_trace, max_peaks_idx, min_peaks_idx, i+1, 
                                    bpm, bpm_std, timing_irregularity, upstroke_time, amplitude)
            calcium_traces.append(ctrace)
            valid_regions.append(ctrace.region)  # Add region number to the list of regions


        # Tissue outputs
        if tissue_calcium_trace.bpm != 0:
            header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'upstroke time', 'amplitude']
            fields = np.array([[sample, tissue_calcium_trace.bpm, tissue_calcium_trace.bpm_std, 
                                tissue_calcium_trace.timing_irregularity, synchronicity,
                                tissue_calcium_trace.upstroke_time, tissue_calcium_trace.amplitude]])
            np.savetxt(f'{path}/{sample}_output.csv', 
                    fields, 
                    delimiter=',', fmt='%s', header=','.join(header), comments='')

            # Save outputs for the tissue
            all_outputs.append(fields)
        else:
            all_outputs.append(np.array([[sample, '', '', '', synchronicity, '', '']]))

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
        clean_traces = np.array([t.trace for t in calcium_traces])
        time = np.arange(len(tissue_calcium_trace.trace)) / framerate  # Time in seconds
        traces_raw = [time]
        traces_raw += [tissue_calcium_trace.trace] if is_one_region else [np.zeros(len(time))]
        traces_raw += [trace for trace in filtered_traces]
        traces_raw = np.vstack(traces_raw).T

        np.savetxt(f'{path}/{sample}_raw_output.csv',
                    traces_raw, 
                    delimiter=',', fmt='%s', header='Time,' + ','.join(['Tissue'] + [f'Region {i+1}' for i in range(len(calcium_traces))]), comments='')

        # Save times
        processing_times.append(timer.time() - start)

        print('done.\n')
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        failed_analyses.append(fname)

# Save all outputs to a single file
if all_outputs:
    all_outputs = np.vstack(all_outputs)
    header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'upstroke time', 'amplitude']
    np.savetxt(f'{selected_folder}/all_samples_output.csv', 
               all_outputs, 
               delimiter=',', fmt='%s', header=','.join(header), comments='')

print(f'Processing completed for {len(filenames)} files.')
print(f'Total processing time: {sum(processing_times):.2f} seconds.')
print(f'Average processing time per file: {np.mean(processing_times):.2f} seconds.')
print(f'Failed analyses: {len(failed_analyses)} files.')
if failed_analyses:
    print('Failed files:', failed_analyses)