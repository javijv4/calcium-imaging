#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/16 18:18:09

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import cv2

from scipy.io import loadmat
from scipy.spatial import KDTree

from skimage import filters, morphology, measure, draw, transform, exposure

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, SpanSelector
from matplotlib import cm

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Loading Data
def load_data(fname, videothresh=None, fix_cut=True):
    print(f"Loading data from {fname}")

    import h5py
    def is_mat73(fname):
        """Check whether the .mat file is in v7.3 (HDF5) format."""
        with open(fname, 'rb') as f:
            header = f.read(128)
        return b'HDF5' in header

    if is_mat73(fname):
        # print("Detected v7.3+ .mat file — using h5py.")
        with h5py.File(fname, 'r') as f:
            # Mimic ['data'][0][0] from loadmat
            data_ref = f['data'][()][0, 0]
            data_group = f[data_ref]

            images = []
            for i in range(len(data_group)):
                # Each entry is a reference to a matrix, so dereference and grab [0]
                ref = data_group[str(i)][0]
                image = f[ref][()]
                images.append(image)
    else:
        # print("Detected v7 or earlier .mat file — using scipy.io.loadmat.")
        mat = loadmat(fname, variable_names=['data']) 
        data_struct = mat['data'][0][0]

        images = []
        for i in range(len(data_struct)):
            images.append(data_struct[i][0])

    data = np.dstack(images)

    if fix_cut:
        data = fix_weird_cut(data)

    data = data[1:-1, 1:-1]  # Remove border pixels

    if videothresh is None:
        videothresh = select_region(data)

    data = data[:, :, videothresh[0]:videothresh[1]]

    return data

def load_data_scipy(fname, videothresh=None, fix_cut=True):
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

# Data rotation options
def rotate_data_cv2(data, mask):
    props = measure.regionprops(mask.astype(int))

    if len(props) == 0:
        print("No regions found in the mask.")
        return mask, data

    # Orientation and centroid
    orientation = props[0].orientation  # Radians
    centroid = props[0].centroid
    angle_deg = -np.degrees(orientation)
    h, w = mask.shape
    center = (int(centroid[1]), int(centroid[0]))  # (x, y)

    # Rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # === Rotate mask ===
    rotated_mask = cv2.warpAffine(mask.astype(int), rot_mat, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=-1)

    # Dilate padded regions
    pad_mask = rotated_mask == -1
    pad_mask = morphology.binary_dilation(pad_mask)
    rotated_mask[pad_mask] = -1  # Re-assign after dilation

    # Crop: keep rows without abrupt transitions
    diff = np.abs(np.diff(rotated_mask, axis=1))
    diff_vals = np.max(diff, axis=1)
    keep_rows = np.where(diff_vals < 2)[0]

    rotated_mask = rotated_mask[keep_rows, :]
    rotated_mask_bool = np.isclose(rotated_mask, 1) # rotated_mask == 1

    # === Rotate data ===
    rotated_data = np.empty((data.shape[0], len(keep_rows), w), dtype=data.dtype)
    for i in range(data.shape[0]):
        rotated_frame = cv2.warpAffine(data[i, :, :], rot_mat, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        rotated_data[i, :, :] = rotated_frame[keep_rows, :]

    return rotated_data, rotated_mask_bool

def rotate_data(data, mask):
    # Find the major axis of the mask and rotate the mask
    props = measure.regionprops(mask[:,:,0].astype(int))

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

# Automated thresholding
def divide_tissue_in_regions(mask, nx=20, ny=5):
    if mask.ndim == 3:
        mask = mask[:, :,0]

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

# GUI for manual thresholding
def find_tissue_regions_interactively(max_data, tissue_mask):
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial threshold value
    threshold_value = filters.threshold_otsu(max_data[tissue_mask > 0])
    binary_mask = max_data > threshold_value
    binary_mask[tissue_mask == 0] = 0

    # Display data
    ax.imshow(max_data, cmap='gray', vmin=0, vmax=np.max(max_data))

    # Display the initial binary mask with transparency
    masked_binary_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
    img = ax.imshow(masked_binary_mask, cmap='viridis', vmin=0, vmax=1, alpha=0.5)
    ax.set_title('Adjust the threshold using the slider below - Exit window when done')

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

    return final_threshold
    # binary_mask = max_data > final_threshold
    # binary_mask[tissue_mask == 0] = 0

    # # Apply morphological operations
    # binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(5))

    # # Grab regions
    # regions = measure.label(binary_mask)  # Label connected components
    
    # return regions

def apply_threshold(final_threshold, data, tissue_mask):
    if data.ndim != 2:
        max_data = np.max(data, axis=2)
    else:
        max_data = data
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]

    binary_mask = max_data > final_threshold
    binary_mask[tissue_mask == 0] = 0

    # Apply morphological operations
    binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(5))

    # Grab regions
    regions = measure.label(binary_mask)  # Label connected components
    
    return regions

# Alternate Method
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    # Ensure image is 3-channel RGB
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.astype(np.uint8)

    image_rgb = (image_rgb * 0.5).astype(np.uint8)

    # Create a color mask
    mask_color = np.zeros_like(image_rgb, dtype=np.uint8)
    mask_color[mask > 0] = color

    # Alpha blend only where mask is active
    overlaid = image_rgb.copy()
    mask_indices = mask > 0
    overlaid[mask_indices] = (
        alpha * mask_color[mask_indices] + (1 - alpha) * image_rgb[mask_indices]
    ).astype(np.uint8)

    return overlaid

def threshold_gui(image_array, mask=None):
    if image_array.ndim != 2:
        raise ValueError("image_array must be 2D")

    if mask is not None:
        if mask.shape != image_array.shape:
            raise ValueError("Mask shape must match image shape")
        mask = mask.astype(bool)
    else:
        mask = np.ones_like(image_array, dtype=bool)

    window = tk.Tk()
    window.title("Adjust Threshold - Hit Done when at Correct Threshold")

    # Convert to PIL image for display
    original_img = Image.fromarray(image_array.astype(np.uint8))
    photo_img = ImageTk.PhotoImage(original_img)

    label = tk.Label(window, image=photo_img)
    label.image = photo_img
    label.pack()

    initial_value = filters.threshold_otsu(image_array[mask > 0])
    threshold_container = {'value': initial_value}

    def update_threshold(val):
        threshold_container['value'] = int(float(val))
        # Create blank image
        new_img_array = np.zeros_like(image_array, dtype=np.uint8)

        # Apply threshold only inside mask
        new_img_array[mask] = np.where(image_array[mask] > threshold_container['value'], 1, 0) #255, 0)

        overlay_img = overlay_mask(image_array, new_img_array)

        new_img = Image.fromarray(overlay_img)
        # new_img = Image.fromarray(new_img_array)
        photo_new_img = ImageTk.PhotoImage(new_img)
        label.configure(image=photo_new_img)
        label.image = photo_new_img

    slider = ttk.Scale(window, from_=0, to=255, orient='horizontal', command=update_threshold)
    slider.set(threshold_container['value'])
    slider.pack(padx=10, pady=10)

    def on_done():
        window.destroy()

    btn = tk.Button(window, text="Done", command=on_done)
    btn.pack(pady=10)

    window.mainloop()
    return threshold_container['value']

# Manual region selection w/ otsu
def find_tissue_regions(data, tissue_mask, threshold_value=None):
    if data.ndim != 2:
        max_data = np.max(data, axis=2)
    else:
        max_data = data
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]

    # Apply Otsu's threshold to the data
    if threshold_value is None:
        # threshold_value = filters.threshold_otsu(max_data[tissue_mask > 0])
        threshold_value = threshold_gui(image_array=max_data, mask=tissue_mask)
        binary_mask = max_data > threshold_value
    else:
        binary_mask = max_data > threshold_value

    binary_mask = morphology.binary_opening(binary_mask, footprint=morphology.disk(5))  # Close small holes
    binary_mask[tissue_mask == 0] = 0  # Ensure we only consider the tissue area defined by the mask

    # Grab regions
    regions = measure.label(binary_mask)  # Label connected components
    
    return regions

# GUI for user region selection
def region_division_gui(image_array, mask, ny, nx):

    if image_array.ndim != 2:
        raise ValueError("image_array must be 2D")

    if mask is not None:
        if mask.shape != image_array.shape:
            raise ValueError("Mask shape must match image shape")
        mask = mask.astype(bool)
    else:
        mask = np.ones_like(image_array, dtype=bool)

    ret_val = {'Selected Option': 0}

    def on_manual_select():
        ret_val['Selected Option'] = 1
        window.destroy()

    def on_auto_select():
        ret_val['Selected Option'] = 2
        window.destroy()

    window = tk.Tk()
    window.title("Choose Region Selection Method")

    # === Frame Setup ===
    frm = ttk.Frame(window, padding=10)
    frm.pack()

    # === Title ===
    title = ttk.Label(frm, text="Select Region Selection Method - Manual selection will open another window", 
                      font=("Arial", 14))
    title.grid(row=0, column=0, columnspan=2, pady=10)

    # === Manual Image: thresholding ===
    thresh = filters.threshold_otsu(image_array[mask])
    manual_array = np.zeros_like(image_array, dtype=np.uint8)
    manual_array[mask] = np.where(image_array[mask] > thresh, 1, 0)#255, 0)
    overlay_img = overlay_mask(image_array, manual_array)
    manual_img = ImageTk.PhotoImage(Image.fromarray(overlay_img))

    manual_label = ttk.Label(frm, text="Manual (threshold-based)")
    manual_label.grid(row=1, column=0)
    manual_canvas = tk.Label(frm, image=manual_img)
    manual_canvas.image = manual_img
    manual_canvas.grid(row=2, column=0, padx=5)

    # === Auto Image: grid division ===
    # Normalize image_array to [0, 255] for display
    base_image = image_array.astype(np.float32)
    base_image = 255 * (base_image - base_image.min()) / (np.ptp(base_image) + 1e-8)
    base_image = base_image.astype(np.uint8)

    # Get labeled regions
    auto_regions = divide_tissue_in_regions(mask, nx=nx, ny=ny)

    # Map labels to colors using a colormap (e.g., viridis)
    colormap = cm.get_cmap('viridis', np.max(auto_regions)+1)
    colored_labels = (colormap(auto_regions)[:, :, :3] * 255).astype(np.uint8)

    # Convert grayscale image to RGB
    base_rgb = np.stack([base_image]*3, axis=-1)

    # Alpha blend label overlay on top of grayscale image
    alpha = 0.4
    blended = (alpha * colored_labels + (1 - alpha) * base_rgb).astype(np.uint8)

    # Convert to Tk image
    auto_img = ImageTk.PhotoImage(Image.fromarray(blended))

    auto_label = ttk.Label(frm, text="Auto (grid-based)")
    auto_label.grid(row=1, column=1)
    auto_canvas = tk.Label(frm, image=auto_img)
    auto_canvas.image = auto_img
    auto_canvas.grid(row=2, column=1, padx=5)

    # === Buttons ===
    btn_manual = ttk.Button(frm, text="Select Manual", command=on_manual_select)
    btn_manual.grid(row=3, column=0, pady=10)

    btn_auto = ttk.Button(frm, text="Select Auto", command=on_auto_select)
    btn_auto.grid(row=3, column=1, pady=10)

    window.mainloop()
    return ret_val['Selected Option']

# Allowing user to select method for region division
def divide_regions_choice(data, mask, nx, ny):
    if data.ndim != 2:
        max_data = np.max(data, axis=2)
    else:
        max_data = data
    if mask.ndim == 3:
        tissue_mask = mask[:, :, 0]
    else:
        tissue_mask = mask

    max_data = filters.gaussian(max_data, sigma=1)  # Smooth the data

    user_selection = region_division_gui(max_data, tissue_mask, ny=ny, nx=nx)

    if user_selection == 1:
        return find_tissue_regions_interactively(max_data, mask), False
    else:
        return 0, True

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

def confirm_mask(mask, data):
    from tkinter import Tk, Label, Button
    from PIL import Image, ImageTk
    
    use_mask = None

    def submit_yes():
        nonlocal use_mask
        use_mask = True
        window.destroy()
    
    def submit_no():
        nonlocal use_mask
        use_mask = False
        window.destroy()


    window = Tk()
    window.title("Confirm Mask")


    # Create an RGB image from the data for visualization
    if data.ndim == 2:
        base_img = data
    else:
        base_img = data[:, :, 0]
    base_img = (base_img - np.min(base_img)) / (np.ptp(base_img) + 1e-8) * 255
    base_img = base_img.astype(np.uint8)
    base_img_rgb = np.stack([base_img]*3, axis=-1)


    # Create a red mask overlay with alpha blending
    mask_alpha = 0.2  # transparency level
    mask_rgb = np.zeros_like(base_img_rgb)
    mask_rgb[..., 0] = 255  # Red channel


    overlay = base_img_rgb.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (mask_alpha * mask_rgb[mask_bool] + (1 - mask_alpha) * base_img_rgb[mask_bool]).astype(np.uint8)


    img = Image.fromarray(overlay)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img 
    panel.pack()


    Label(window, text="Is the Mask Correct?").pack(pady=(10, 0))
    Button(window, text='Yes', command=submit_yes).pack()
    Button(window, text='No', command=submit_no).pack()

    window.mainloop()


    return use_mask

def get_tissue_mask(data):
        
    # if is_one_region:
    if data.ndim == 3:          # We only care about the first frame
        data = data[:, :, 0]

    # Improve data for the mask
    data = exposure.equalize_hist(data)  # Histogram equalization
    data = filters.gaussian(data, sigma=5)  # Smooth the data

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

    # Fill any holes in the mask
    mask_area = np.sum(binary_mask)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=mask_area / 5)

    mask_correct = confirm_mask(binary_mask.astype(np.uint8) * 255, data)

    # else:
    if not mask_correct:
        _, ax = plt.subplots()
        selector = MaskSelector(ax, data)
        plt.show()

        binary_mask = selector.mask

    return binary_mask


def evaluate_regional_intensities(data, regions):
    nregions = np.max(regions)
    nframes = data.shape[0]

    intensities = np.zeros((nframes, nregions))

    for i in range(nframes):
        for j in range(nregions):
            frame = data[i]
            intensities[i, j] = np.mean(frame[regions == j+1])

    return intensities