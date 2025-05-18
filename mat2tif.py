import os
import numpy as np
from scipy.io import loadmat
from skimage.io import imsave

import imutils as imu

def mat_to_tif(input_path, output_path):
    """
    Converts a .mat file containing image data to a .tif file.

    Parameters:
        input_path (str): Path to the input .mat file.
        output_path (str): Path to save the output .tif file.
    """
    # Load the .mat file
    image_data = imu.load_data(input_path, videothresh = (200, 500) )
    image_data = np.swapaxes(image_data, 0, 2)
        
    # Save the image data as a .tif file
    imsave(output_path, image_data.astype(np.float32))
    print(f"Saved .tif file to {output_path}")


def mat_to_tif_warped(input_path, output_path):
    """
    Converts a .mat file containing image data to a .tif file.

    Parameters:
        input_path (str): Path to the input .mat file.
        output_path (str): Path to save the output .tif file.
    """
    # Load the .mat file
    image_data = loadmat(f'{input_path}')['warped_data']
    image_data = np.swapaxes(image_data, 0, 2)
        
    # Save the image data as a .tif file
    imsave(output_path, image_data.astype(np.float32))
    print(f"Saved .tif file to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_mat_file = "test_data2/fibers_iPSCCF_day7-06.mat"  # Replace with your .mat file path
    output_tif_file = "test_data2/fibers_iPSCCF_day7-06.tif"  # Replace with your desired .tif file path
    
    if not os.path.exists(input_mat_file):
        print(f"Input file {input_mat_file} does not exist.")
    else:
        if 'warped' in input_mat_file:
            mat_to_tif_warped(input_mat_file, output_tif_file)
        else:
            mat_to_tif(input_mat_file, output_tif_file)