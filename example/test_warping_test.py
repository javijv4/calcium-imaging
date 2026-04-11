
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from imutils import normalize_image
import imregistration as imreg
import imutils as imu

videothresh = (100, 101)

np.random.seed(0)

#%%
def warp_stack_og(data_og, displacements):
    """Apply per-frame displacements to align frames to frame 0."""
    tlen, h, w = data_og.shape
    i = np.arange(h)
    j = np.arange(w)
    I, J = np.meshgrid(i, j)
    IJ = np.vstack([I.ravel(), J.ravel()]).T

    warped_data = np.zeros_like(data_og)

    print("Warping data")
    for t in tqdm(range(tlen)):
        disp = np.vstack(
            [
                displacements[t, :, :, 0].ravel(),
                displacements[t, :, :, 1].ravel(),
            ]
        ).T
        ij = IJ - disp.astype(int)
        ij[:,0] = np.clip(ij[:,0], 0, w - 1)
        ij[:,1] = np.clip(ij[:,1], 0, h - 1)

        warped_data[t, :, :] = data_og[t, ij[:, 1], ij[:, 0]].reshape(h, w)

    vmin = data_og.min()
    vmax = data_og.max()
    warped_data = normalize_image(warped_data) * (vmax - vmin) + vmin

    return warped_data

def warp_stack(data_og, displacements):
    """Apply per-frame displacements to align frames to frame 0.

    Matches ``warp_stack_og`` / ``imregistration.warp_stack``: default
    ``meshgrid(i, j)`` is ``indexing='xy'``, so the raveled coordinate pairs
    correspond to ``data_og[t, ij[:, 1], ij[:, 0]]`` then ``reshape(h, w)``.
    """
    tlen, h, w = data_og.shape
    i = np.arange(h)
    j = np.arange(w)
    J, I = np.meshgrid(i, j, indexing='ij')
    IJ = np.vstack([I.ravel(), J.ravel()]).T

    warped_data = np.zeros_like(data_og)

    print("Warping data")
    for t in tqdm(range(tlen)):
        i = I.ravel() - displacements[t, :, :, 0].ravel().astype(int)
        j = J.ravel() - displacements[t, :, :, 1].ravel().astype(int)
        i = np.clip(i, 0, w - 1)
        j = np.clip(j, 0, h - 1)
        warped_data[t, :, :] = data_og[t, j, i].reshape(h, w)

    vmin = data_og.min()
    vmax = data_og.max()
    # warped_data = normalize_image(warped_data) * (vmax - vmin) + vmin

    return warped_data


def test_data(data, displacements):

    data_og = data.copy()
    warped_data_og = warp_stack_og(data_og, displacements)
    warped_data = warp_stack(data_og, displacements)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(data[0, :, :])
    axs[0].set_title('Original')
    axs[1].imshow(warped_data[0, :, :])
    axs[1].set_title('New Warped')
    axs[2].imshow(warped_data_og[0, :, :])
    axs[2].set_title('Old Warped')
    plt.show()

    print(np.allclose(warped_data, warped_data_og))

    return warped_data, warped_data_og
    
    
size_x = 5
size_y = 5
x = np.arange(size_x)
y = np.arange(size_y)
displacements = np.dstack(np.meshgrid(x, y, indexing='ij'))
displacements = displacements.reshape(1, size_x, size_y, 2)
data = displacements[:,:,:,0]*displacements[:,:,:,1]
displacements = np.zeros_like(displacements)
displacements[0, :size_x, :size_x] = np.random.rand(1, size_x, size_x, 2) * size_x - size_x/2
# displacements = np.abs(displacements)
displacements_square = displacements.copy()
warped_data_square, warped_data_og_square = test_data(data, displacements)

size_x = 5
size_y = 10
x = np.arange(size_x)
y = np.arange(size_y)
displacements = np.dstack(np.meshgrid(x, y, indexing='ij'))
displacements = displacements.reshape(1, size_x, size_y, 2)
data = displacements[:,:,:,0]*displacements[:,:,:,1]
displacements = np.zeros_like(displacements)
displacements[0, :size_x, :size_x] = displacements_square

warped_data, warped_data_og = test_data(data, displacements)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(warped_data[0, :size_x, :size_x])
axs[0].set_title('New Warped')
axs[1].imshow(warped_data_square[0, :size_x, :size_x])
axs[1].set_title('Old Warped')
axs[2].imshow(warped_data[0, :size_x, :size_x] - warped_data_square[0, :size_x, :size_x])
axs[2].set_title('Difference')
plt.show()
print(np.allclose(warped_data[0, :size_x, :size_x], warped_data_og_square[0, :size_x, :size_x]))