#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/08/16

@author: Javiera Jilberto Vallejos 
'''

import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import io
from skimage import io as skio

import imregistration as imreg
import imutils as imu
import calcium_analysis as ca
import plotutils as pu

from tkinter import Tk, filedialog
import glob

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
        for fname in glob.glob(os.path.join(folder, f'*{ext}')):
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
imreg.NCORES = 8                # Number of cores for registration
threshold_value = 0             # If 0, otsu thresholding is used
width_factor = 1/10
plot_all_traces = True
plot_unwarped_traces = True     # Also save all_traces using the non-warped stack
rotate_orientation = None
fix_cut = False

# Select a folder using the GUI
selected_folder = select_folder()

# Get all stack files in the selected folder
filenames = []
# filenames = ['test_data_richard/TTN-mSca_GCaMP_488_50fps_40x_1002.tif']
if len(filenames) == 0:
    filenames = list_stack_files(selected_folder)

processing_times = []
failed_analyses = []
all_outputs = []
for fname in filenames:
    # try:
    start = timer.time()  # Start timing the analysis

    # Dealing with paths
    sample = os.path.splitext(os.path.basename(fname))[0]
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

    # Load registration
    if os.path.exists(f'{path}/{sample}_warped.mat'):
        print('Loading warped images...')
        warped_data = io.loadmat(f'{path}/{sample}_warped.mat')['warped_data']
        imu.save_data(f'{path}/{sample}_warped.tif', warped_data)
    else:
        print('Warped images not found. Please run the registration script first.')
        continue

    # Rotate the data such that the tissue is vertical
    if rotate_orientation is not None:
        print('Rotating data...')
        warped_data, mask = imu.rotate_data_cv2(warped_data, mask, target_orientation=rotate_orientation)
        first_frame = warped_data[0]
        skio.imsave(f'{path}/{sample}_tissue_mask_rotated.tif', mask.astype(np.uint8) * 255)
        skio.imsave(f'{path}/{sample}_warped_tissue_rotated.tif', warped_data.astype(np.int16),
                    check_contrast=False)
    else:
        first_frame = warped_data[0]

    # Divide the tissue in regions
    if is_one_region:
        regions = imu.divide_tissue_in_regions(mask=mask, nx=region_params[0], ny=region_params[1], horizontal=True)
    else:
        regions = imu.apply_threshold(final_threshold=region_params[0], data=first_frame, tissue_mask=mask)

    # Evaluate intensities in the whole tissue
    print("Analyzing tissue intensities...")
    if is_one_region:
        tissue_trace = imu.evaluate_regional_intensities(warped_data, mask.astype(int))[:, 0]

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
        filtered_traces.append(filtered_trace)
        max_peaks.append(max_peaks_idx)
        min_peaks.append(min_peaks_idx)

    # Plot all traces
    if plot_all_traces:
        ntraces = len(filtered_traces)
        fig, axs = plt.subplots(ntraces, 1, figsize=(10, 2 * ntraces), sharex=True)
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

    # Same region-trace plot from the non-warped stack
    filtered_raw_traces = None
    if plot_unwarped_traces:
        print('Plotting unwarped traces...')
        raw_data = imu.load_data(fname, videothresh=videothresh, fix_cut=fix_cut)
        raw_traces = imu.evaluate_regional_intensities(raw_data, regions)
        filtered_raw_traces = []
        for trace in raw_traces.T:
            filtered_trace, _, _ = ca.analyze_trace_fft(trace, framerate=framerate, width_factor=width_factor)
            filtered_raw_traces.append(filtered_trace)
        filtered_raw_traces = np.array(filtered_raw_traces)
        _, fig_unwarped = pu.plot_regions_traces(raw_data[0], regions, filtered_raw_traces, framerate=framerate)
        fig_unwarped.savefig(f'{path}/{sample}_all_traces_unwarped.png', dpi=300, bbox_inches='tight')
        plt.close(fig_unwarped)

    # Synchronicity (mean pairwise correlation of regional traces)
    synchronicity = np.mean(np.corrcoef(filtered_traces))
    if filtered_raw_traces is not None:
        synchronicity_unwarped = np.mean(np.corrcoef(filtered_raw_traces))
    else:
        synchronicity_unwarped = ''
    print(f'Synchronicity (warped): {synchronicity:.6f}')
    if filtered_raw_traces is not None:
        print(f'Synchronicity (unwarped): {synchronicity_unwarped:.6f}')

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
        ctrace = ca.CalciumTrace(filtered_trace, max_peaks_idx, min_peaks_idx, i + 1,
                                bpm, bpm_std, timing_irregularity, upstroke_time, amplitude)
        calcium_traces.append(ctrace)
        valid_regions.append(ctrace.region)

    # Tissue outputs
    header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'synchronicity unwarped', 'upstroke time', 'amplitude']
    if tissue_calcium_trace.bpm != 0:
        fields = np.array([[sample, tissue_calcium_trace.bpm, tissue_calcium_trace.bpm_std,
                            tissue_calcium_trace.timing_irregularity, synchronicity, synchronicity_unwarped,
                            tissue_calcium_trace.upstroke_time, tissue_calcium_trace.amplitude]])
        np.savetxt(f'{path}/{sample}_output.csv',
                fields,
                delimiter=',', fmt='%s', header=','.join(header), comments='')

        all_outputs.append(fields)
    else:
        all_outputs.append(np.array([[sample, '', '', '', synchronicity, synchronicity_unwarped, '', '']]))

    # Region outputs
    outputs = []
    for i, ctrace in enumerate(calcium_traces):
        header = ['Sample Name', 'Region', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'synchronicity unwarped', 'upstroke time', 'amplitude']
        outputs.append([sample, ctrace.region, ctrace.bpm, ctrace.bpm_std,
                        ctrace.timing_irregularity, synchronicity, synchronicity_unwarped,
                        ctrace.upstroke_time, ctrace.amplitude])

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
    # except Exception as e:
        # print(f"Error processing {fname}: {e}")
        # failed_analyses.append(fname)

# Save all outputs to a single file
if all_outputs:
    all_outputs = np.vstack(all_outputs)
    header = ['Sample Name', 'bpm', 'bpm std', 'timing irreg', 'synchronicity', 'synchronicity unwarped', 'upstroke time', 'amplitude']
    np.savetxt(f'{selected_folder}/all_samples_output.csv',
               all_outputs,
               delimiter=',', fmt='%s', header=','.join(header), comments='')

print(f'Processing completed for {len(filenames)} files.')
print(f'Total processing time: {sum(processing_times):.2f} seconds.')
print(f'Average processing time per file: {np.mean(processing_times) if processing_times else 0:.2f} seconds.')
print(f'Failed analyses: {len(failed_analyses)} files.')
if failed_analyses:
    print('Failed files:', failed_analyses)
