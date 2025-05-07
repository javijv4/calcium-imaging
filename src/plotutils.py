#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/16 18:22:25

@author: Javiera Jilberto Vallejos 
'''

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable


def overlap_images(image1, image2):
    # Normalize the frames to [0, 1] before creating the RGB image
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    image2 = (image2 - image2.min()) / (image2.max() - image2.min())

    # Create an RGB image to show the overlap
    rgb = np.zeros((*image1.shape, 3))
    rgb[:,:,0] = image1  # Red channel
    rgb[:,:,1] = image2  # Green channel

    return rgb


def plot_registration_results(fixed, moving, warped_img, displacement_field):
    # Grab two frames and plot them using subplots with ax
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the first frame
    ax[0, 0].imshow(moving, cmap='gray')
    ax[0, 0].set_title('Frame 400')

    # Plot the second frame
    ax[0, 1].imshow(fixed, cmap='gray')
    ax[0, 1].set_title('Frame 420')

    # Normalize the frames to [0, 1] before creating the RGB image
    rgb = overlap_images(moving, fixed)

    # Plot the overlap
    ax[0, 2].imshow(rgb)
    ax[0, 2].set_title('Overlap')

    # Plot the displacement field using quiver
    Y, X = np.mgrid[0:displacement_field.shape[0], 0:displacement_field.shape[1]]
    every_nth = 50
    Y = Y[::every_nth, ::every_nth]
    X = X[::every_nth, ::every_nth]
    U = displacement_field[::every_nth, ::every_nth, 0]
    V = displacement_field[::every_nth, ::every_nth, 1]
    ax[1, 0].quiver(X, Y, U, V, color='red')
    ax[1, 0].set_title('Displacement Field')

    # Plot the warped image
    ax[1, 1].imshow(warped_img, cmap='gray')
    ax[1, 1].set_title('Warped Image')

    # Normalize the warped image and the moving image before creating the RGB image
    warped_rgb = overlap_images(warped_img, fixed)

    # Plot the overlap of the warped image and the moving image
    ax[1, 2].imshow(warped_rgb)
    ax[1, 2].set_title('Warped Overlap')


    # Calculate and annotate the norm of the difference in intensity
    norm_diff_init = np.linalg.norm(fixed - moving)
    norm_diff = np.linalg.norm(fixed - warped_img)


    ax[0, 2].set_title(f'Initial difference: {norm_diff_init:.2f}', 
                horizontalalignment='center', verticalalignment='center', fontsize=12)

    ax[1, 2].set_title(f'Final difference: {norm_diff:.2f}', 
                horizontalalignment='center', verticalalignment='center', fontsize=12)

    # Remove axis for all subplots
    for a in ax.ravel():
        a.axis('off')

    plt.show()


def plot_quiver_displacement(data, displacement_field):
    nframes = data.shape[2]

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial frame
    frame_idx = 0
    im = ax.imshow(data[:, :, frame_idx], cmap='gray')

    # Quiver plot for displacement field
    every_nth = 50
    scale_factor = 300  # Adjust this value to scale the arrows
    Y, X = np.mgrid[0:data.shape[0]:every_nth, 0:data.shape[1]:every_nth]
    quiver = ax.quiver(X, Y, 
                       -displacement_field[::every_nth, ::every_nth, frame_idx, 0], 
                       displacement_field[::every_nth, ::every_nth, frame_idx, 1], 
                       color='r', scale=scale_factor)

    # Slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, nframes-1, valinit=frame_idx, valstep=1)

    # Update function
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(data[:, :, frame_idx])
        quiver.set_UVC(-displacement_field[::every_nth, ::every_nth, frame_idx, 0], 
                       displacement_field[::every_nth, ::every_nth, frame_idx, 1])
        fig.canvas.draw_idle()

    # Call update function on slider value change
    slider.on_changed(update)

    plt.show()


def plot_points_displacement(points, data, displacement_fields):
    nframes = data.shape[2]

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial frame
    frame_idx = 0
    im = ax.imshow(data[:, :, frame_idx], cmap='gray')
    [ax.plot(point[1], point[0], 'b.')[0] for point in points]
    point_plots = [ax.plot(point[1], point[0], 'r.')[0] for point in points]

    # Slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, nframes-1, valinit=frame_idx, valstep=1)

    # Update function
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(data[:, :, frame_idx])
        for i, point in enumerate(points):
            dispx = displacement_fields[point[0], point[1], frame_idx, 1]
            dispy = displacement_fields[point[0], point[1], frame_idx, 0]
            disp = np.array([dispx, dispy])
            new_point = point - disp
            point_plots[i].set_data([new_point[1]], [new_point[0]])
        fig.canvas.draw_idle()

    # Call update function on slider value change
    slider.on_changed(update)

    plt.show()

def plot_regions_intensity(data, regions, intensities):
    if data.ndim == 3:
        data0 = data[:, :, 0]  # Use the first frame for plotting regions
    else:
        data0 = data

    nregions = np.max(regions)

    # Plot regions and intensity traces
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # Plot data with regions overlay
    ax0 = fig.add_subplot(gs[0])
    region_colors = plt.cm.get_cmap('jet', nregions)
    colored_regions = np.zeros((*regions.shape, 4))  # Add alpha channel for transparency
    for i in range(nregions):
        colored_regions[regions == i+1, :3] = region_colors(i)[:3]
        colored_regions[regions == i+1, 3] = 0.5  # Set alpha for regions

    colored_regions[regions == 0, 3] = 0  # Fully transparent for regions == 0

    ax0.axis('off')
    ax0.imshow(data0, cmap='gray')
    ax0.imshow(colored_regions)
    ax0.set_title('Segmented Regions')

    # Plot intensity traces
    ax1 = fig.add_subplot(gs[1])
    for i in range(nregions):
        ax1.plot(intensities[:, i], label=f'Region {i+1}', color=region_colors(i))

    ax1.set_title('Intensity Traces')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Intensity')

    plt.tight_layout()


def plot_regions_traces_interactive(data, regions, calcium_traces, framerate=1):
    traces = np.array([trace.trace for trace in calcium_traces])
    traces_regions = np.array([trace.region for trace in calcium_traces])

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    # Left subplot: show the warped data with regions overlay
    warped_ax = axes[0]
    warped_im = warped_ax.imshow(data[:, :, 0], cmap='gray', aspect='auto')
    warped_ax.set_title('Warped Data with Regions')
    warped_ax.axis('off')

    # Overlay regions with transparency
    regions_overlay = warped_ax.imshow(
        regions, cmap='jet', alpha=0.2, aspect='auto', norm=Normalize(vmin=1, vmax=np.max(regions))
    )
    regions_overlay.set_alpha(np.where(regions == 0, 0, 0.2))  # Fully transparent where regions == 0

    # Right subplot: plot the traces
    traces_ax = axes[1]
    time = np.arange(traces.shape[1]) / framerate
    cmap = plt.cm.get_cmap('jet', np.max(regions))  # Use the same colormap as the regions
    for trace, region in zip(traces, traces_regions):
        color = to_rgba(cmap(region))  # Get the color corresponding to the region
        traces_ax.plot(time, trace, alpha=1.0, color=color)
    vertical_line = traces_ax.axvline(x=0, color='0.5', linestyle='-')
    traces_ax.set_title('Traces')
    traces_ax.set_xlabel('Time (s)')
    traces_ax.set_ylabel('Intensity')

    # Add a slider for the timestep
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Timestep', 0, data.shape[-1] - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        timestep = int(slider.val)
        warped_im.set_data(data[:, :, timestep])
        vertical_line.set_xdata([timestep / framerate])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def plot_regions_traces(data, regions, calcium_traces, framerate=1):
    traces = np.array([trace.trace for trace in calcium_traces])
    traces_regions = np.array([trace.region for trace in calcium_traces])

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    # Left subplot: show the warped data with regions overlay
    warped_ax = axes[0]
    warped_im = warped_ax.imshow(data[:, :, 0], cmap='gray', aspect='auto')
    warped_ax.set_title('Warped Data with Regions')
    warped_ax.axis('off')

    # Overlay regions with transparency
    regions_overlay = warped_ax.imshow(
        regions, cmap='jet', alpha=0.2, aspect='auto', norm=Normalize(vmin=1, vmax=np.max(regions))
    )
    regions_overlay.set_alpha(np.where(regions == 0, 0, 0.5))  # Fully transparent where regions == 0

    # Right subplot: plot the traces
    traces_ax = axes[1]
    time = np.arange(traces.shape[1]) / framerate
    cmap = plt.cm.get_cmap('jet', np.max(regions))  # Use the same colormap as the regions
    for trace, region in zip(traces, traces_regions):
        color = to_rgba(cmap(region))  # Get the color corresponding to the region
        traces_ax.plot(time, trace, alpha=1.0, color=color)
    traces_ax.set_title('Traces')
    traces_ax.set_xlabel('Time (s)')
    traces_ax.set_ylabel('Intensity')


def save_png_points(points, data, displacement_fields, output_dir='pngs'):
    nframes = data.shape[2]

    # # Create directory for png images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots()

    ax.axis('off')

    # Initial frame
    frame_idx = 0
    im = ax.imshow(data[:, :, frame_idx], cmap='gray')
    point_plots = [ax.plot(point[1], point[0], 'r.')[0] for point in points]

    # Function to update the plot for each frame
    def update_frame(frame_idx):
        im.set_data(data[:, :, frame_idx])
        for i, point in enumerate(points):
            dispx = displacement_fields[point[0], point[1], frame_idx, 1]
            dispy = displacement_fields[point[0], point[1], frame_idx, 0]
            disp = np.array([dispx, dispy])
            new_point = point - disp
            point_plots[i].set_data(new_point[1], new_point[0])
        return [im] + point_plots

    # Loop over each frame and save the plot as an image
    for frame_idx in range(nframes):
        update_frame(frame_idx)
        fig.canvas.draw()
        image_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        fig.savefig(image_path, bbox_inches='tight', pad_inches=0)

    plt.close(fig)
