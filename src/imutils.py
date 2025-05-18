#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/16 18:18:09

@author: Javiera Jilberto Vallejos 
'''

import numpy as np

from scipy.io import loadmat
from scipy.spatial import KDTree

from skimage import filters, morphology, measure, draw, transform

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, SpanSelector



def load_data(fname, videothresh=None, fix_cut=True):
    print(f"Loading data from {fname}")
    data = loadmat(fname)['data'][0][0]

    # Dealing with the data
    images = []
    for i in range(len(data)):
        images.append(data[i][0])

    data = np.dstack(images)
    data = fix_weird_cut(data)
    data = data[1:-1,1:-1]        # There are some border pixels with high values

    if videothresh is None:
        videothresh = select_region(data)
    data = data[:, :, videothresh[0]:videothresh[1]]

    return data


def select_region(data):
    sum_values = np.sum(data, axis=(0, 1))

    fig, ax = plt.subplots(1)

    ax.plot(sum_values, 'k')
    ax.set_title('Press left mouse button and drag to select a region in the top graph\n'
                 'Close the window to continue')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Sum of pixel values')

    selected_xlim = []

    def onselect(xmin, xmax):
        selected_xlim.clear()
        selected_xlim.extend([xmin, xmax])
        fig.canvas.draw_idle()

    span = SpanSelector(
        ax,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:blue"),
        interactive=True,
        drag_from_anywhere=True
    )
    plt.show()

    selected_xlim = [int(x) for x in selected_xlim]
    print(f"Selected region: {selected_xlim}")

    return selected_xlim


def fix_weird_cut(data, cut=512):
    new_data = np.zeros_like(data)
    new_data[:,0:data.shape[1]-cut:,:] = data[:,cut:,:]
    new_data[:,data.shape[1]-cut:,:] = data[:,0:cut,:]
    data = new_data
    return data


def rotate_data(data, mask):
    # Find the major axis of the mask and rotate the mask
    props = measure.regionprops(mask.astype(int))

    if len(props) > 0:
        # Get the orientation of the largest region
        orientation = props[0].orientation  # Angle in radians
        centroid = props[0].centroid

        mask_aux = mask.astype(int)

        # Rotate the mask to align with the major axis
        rotated_mask = transform.rotate(mask_aux, angle=np.degrees(-orientation), center=centroid, mode='constant', preserve_range=True, cval=-1)
        rotated_mask = np.round(rotated_mask).astype(int)

        pad_mask = rotated_mask == -1
        pad_mask = morphology.binary_dilation(pad_mask)

        rotated_mask[pad_mask] = -1

        # Check where to cut
        diff = np.abs(np.diff(rotated_mask, axis=1))
        diff_vals = np.max(diff, axis=1)
        keep = np.where(diff_vals < 2)[0]
        rotated_mask = rotated_mask[keep, :]
        rotated_mask = np.isclose(rotated_mask, 1)

        # Rotate the data to align with the rotated mask
        rotated_data = [transform.rotate(data[:,:,i], angle=np.degrees(-orientation), center=centroid, preserve_range=True) for i in range(data.shape[2])]
        rotated_data = np.dstack([frame[keep, :] for frame in rotated_data])
    else:
        print("No regions found in the mask.")
        rotated_mask = mask
        rotated_data = data

    return rotated_data, rotated_mask


def divide_tissue_in_regions(mask, nx=20, ny=5):
    if mask.ndim == 3:
        mask = mask[:, :,0]

    # Making sure mask is smooth
    mask = morphology.binary_closing(mask, morphology.disk(21))
    mask = filters.gaussian(mask, sigma=20) > 0.5

    # Get center of cells
    xlimits = np.where(np.sum(mask, axis=1)>0)[0]
    xlimits = [xlimits[0], xlimits[-1]]
    L_im_x = np.diff(xlimits)[0]

    x_cells_norm = np.linspace(0, 1, 2*ny+1)[1::2]
    y_cells_norm = np.linspace(0, 1, 2*nx+1)[1::2]

    cont = 0
    xy_cells_im = np.zeros([ny*nx, 2])
    for x in x_cells_norm:
        xlim_dn = int(x*L_im_x)-2 + xlimits[0]
        xlim_up = int(x*L_im_x)+2 + xlimits[0]
        bk_x = np.where(mask[xlim_dn:xlim_up]==1)[1]
        Ly = np.max(bk_x) - np.min(bk_x)


        for y in y_cells_norm:
            xy_cells_im[cont,0] = x*L_im_x + xlimits[0]
            xy_cells_im[cont,1] = y*Ly + np.min(bk_x)
            cont += 1


    # Assign each pixel to a cell
    i = np.arange(mask.shape[0], dtype=float)
    j = np.arange(mask.shape[1], dtype=float)
    i, j = np.meshgrid(i,j)
    ij = np.vstack([i.flatten(),j.flatten()]).T   # Position in pixels


    tree = KDTree(xy_cells_im)
    _, cell = tree.query(ij)


    # Reshaping to an image
    cell = cell.reshape(mask.T.shape).T + 1
    cell[mask==0] = 0

    return cell



def find_tissue_regions_interactively(data, tissue_mask):
    max_data = np.max(data, axis=2)
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial threshold value
    threshold_value = filters.threshold_otsu(max_data)
    binary_mask = max_data > threshold_value
    binary_mask[tissue_mask == 0] = 0

    # Display data
    ax.imshow(max_data, cmap='gray', vmin=0, vmax=np.max(max_data))

    # Display the initial binary mask with transparency
    masked_binary_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
    img = ax.imshow(masked_binary_mask, cmap='viridis', vmin=0, vmax=1, alpha=0.5)
    ax.set_title('Adjust the threshold using the slider below')

    # Slider for threshold adjustment
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = plt.Slider(ax_slider, 'Threshold', np.min(max_data), np.max(max_data), valinit=threshold_value)

    def update(val):
        threshold = slider.val
        binary_mask = max_data > threshold
        masked_binary_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
        img.set_data(masked_binary_mask)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    # Final threshold value after slider adjustment
    final_threshold = slider.val
    binary_mask = max_data > final_threshold
    binary_mask[tissue_mask == 0] = 0

    # Apply morphological operations
    binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(5))

    # Grab regions
    regions = measure.label(binary_mask)  # Label connected components
    
    return regions


def find_tissue_regions(data, tissue_mask, threshold_value=None):
    max_data = np.max(data, axis=2)
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]

    # Apply Otsu's threshold to the data
    if threshold_value is None:
        threshold_value = filters.threshold_otsu(max_data[tissue_mask > 0])
        binary_mask = max_data > threshold_value
    else:
        binary_mask = max_data > threshold_value

    binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(5))  # Close small holes
    binary_mask[tissue_mask == 0] = 0  # Ensure we only consider the tissue area defined by the mask

    # Grab regions
    regions = measure.label(binary_mask)  # Label connected components
    
    return regions



def normalize_image(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    image = (image - min_intensity) / (max_intensity - min_intensity)
    return image

def in_plane_footprint(size):
    footprint = np.ones((size, size, 1))
    return footprint

def get_tissue_centroid(data):
    # # Apply Otsu's threshold to the data
    threshold_value = filters.threshold_otsu(data[:,:,0])
    binary_mask = data[:,:,0] > threshold_value

    ij = np.array(np.where(binary_mask))
    centroid = np.mean(ij, axis=1)
    return centroid.astype(int)



class MaskSelector:

    def __init__(self, ax, img, mask0=None):
        self.canvas = ax.figure.canvas
        self.img = img
        self.mask = mask0
        self.ax = ax
        self.verts = []

        self.poly = PolygonSelector(ax, self.onselect, props=dict(color='r', linestyle='-', linewidth=2, alpha=0.5))

        self.ax.imshow(self.img, cmap='gray')
        self.ax.axis('off')  # Turn off axis for better visualization
        self.ax.set_title('Draw a polygon to select the tissue area and press Enter to confirm\n'
        'Press the "esc" key to start a new polygon.\n'
        'You can also hold the "shift" key to move all vertices or "ctrl" to move a single vertex.\n')

        self.canvas.mpl_connect('key_press_event', self.on_key)

    def onselect(self, verts):
        self.reset()
        self.verts = verts
        self.canvas.draw_idle()

    def reset(self):
        if hasattr(self, 'lines'):
            for line in self.lines:
                for l in line:
                    l.remove()
        self.lines = []

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def get_mask(self):
        self.mask = np.zeros(self.img.shape, dtype=bool)
        if not self.verts:
            return self.mask
        
        # Use the polygon vertices to create a mask
        verts = np.fliplr(np.array(self.verts))  # Flip the vertices to match the (row, column) format expected by skimage
        self.mask = draw.polygon2mask(self.img.shape, verts.astype(int))
        
        return self.mask

    def on_key(self, event):
        if event.key == 'enter':
            self.mask = self.get_mask()
            plt.close(self.ax.figure)


def get_tissue_mask(data, is_one_region=False):
        
    if is_one_region:
        if data.ndim == 3:          # We only care about the first frame
            data = data[:, :, 0]

        # # Apply Otsu's threshold to the data
        threshold_value = filters.threshold_otsu(data)
        binary_mask = data > threshold_value

        binary_mask = morphology.binary_closing(binary_mask, footprint=morphology.disk(5))
        binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(10))
        binary_mask = filters.gaussian(binary_mask, sigma=10) > 0.5

        # Keep only the largest object
        labeled_slice = measure.label(binary_mask)
        regions = measure.regionprops(labeled_slice)
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            largest_mask = labeled_slice == largest_region.label
            binary_mask = largest_mask

    
    else:
        max_data = np.max(data, axis=2)
        _, ax = plt.subplots()
        selector = MaskSelector(ax, max_data)
        plt.show()

        binary_mask = selector.mask

    return binary_mask


def evaluate_regional_intensities(data, regions):
    nregions = np.max(regions)
    nframes = data.shape[2]

    intensities = np.zeros((nframes, nregions))

    for i in range(nframes):
        for j in range(nregions):
            intensities[i, j] = np.mean(data[regions == j+1, i])

    return intensities