#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/16 18:28:09

@author: Javiera Jilberto Vallejos 
'''


import numpy as np
from scipy.signal import savgol_filter, find_peaks
from dataclasses import dataclass


def analyze_trace(trace):
    # Filter trace using Savitzky-Golay filter (line 163)
    filt_trace = savgol_filter(trace, window_length=21, polyorder=7)

    # Find max peaks 
    max_signal = np.max(filt_trace)
    min_signal = np.min(filt_trace)
    prominence = (max_signal - min_signal) * 0.5
    max_peaks_idx, _ = find_peaks(filt_trace, prominence=prominence)
    if len(max_peaks_idx) <= 3:         # If there are not enough peaks, return empty lists
        return filt_trace, [], []

    # Find min peaks
    min_peaks_idx, _ = find_peaks(-filt_trace, prominence=prominence)

    # Fix the drift in the signal
    min_peaks_value = filt_trace[min_peaks_idx]
    correction_value_peaks = min_peaks_value[0] - min_peaks_value
    correction_value_frames = np.interp(np.arange(len(filt_trace)), min_peaks_idx, correction_value_peaks)
    filt_trace = filt_trace + correction_value_frames

    return filt_trace, max_peaks_idx, min_peaks_idx


def trace_outputs(trace, max_peaks_idx, min_peaks_idx, fps):
    # Crop the trace such that it starts and ends in a valley
    crop_trace = trace[min_peaks_idx[0]:min_peaks_idx[-1]+1]
    offset = min_peaks_idx[0]

    if max_peaks_idx[0] < min_peaks_idx[0]:
        max_peaks_idx = max_peaks_idx[1:]
    if max_peaks_idx[-1] > min_peaks_idx[-1]:
        max_peaks_idx = max_peaks_idx[:-1]
    max_peaks_idx = max_peaks_idx - offset 
    min_peaks_idx = min_peaks_idx - offset

    max_peaks_time = max_peaks_idx / fps  # Convert to time in seconds
    min_peaks_time = min_peaks_idx / fps  # Convert to time in seconds
    min_peaks_value = crop_trace[min_peaks_idx]  # Get min peaks values

    cycle_duration = np.diff(max_peaks_time)  # Calculate cycle duration

    # Outputs (line 174-177)
    nframes = crop_trace.shape[0]
    time = np.arange(nframes) / fps  # Time in seconds
    total_time = np.max(time)  # Total time in seconds
    bpm = len(max_peaks_idx)/total_time*60                  
    bpm_std = np.std(cycle_duration)        
    timing_irregularity = bpm_std / np.mean(cycle_duration)  # Calculate timing irregularity

    # Upstroke time 
    upstroke_time_cycles = max_peaks_time - min_peaks_time[:-1]
    upstroke_time = np.mean(upstroke_time_cycles)  # Average upstroke time

    # Amplitude
    amplitude_cycles = crop_trace[max_peaks_idx] - min_peaks_value[:-1]
    amplitude = np.mean(amplitude_cycles)  # Average amplitude

    return bpm, bpm_std, timing_irregularity, upstroke_time, amplitude


def get_synchronicity(traces):
    """
    Calculate the synchronicity of the traces.
    """
    # Calculate the synchronicity of the traces
    synchronicity = np.mean(np.corrcoef(traces.T))

    return synchronicity


@dataclass
class CalciumTrace:
    """
    Class to store the calcium trace information.
    """
    trace: np.ndarray
    max_peaks_idx: np.ndarray
    min_peaks_idx: np.ndarray
    region: int
    bpm: float
    bpm_std: float
    timing_irregularity: float
    upstroke_time: float
    amplitude: float
