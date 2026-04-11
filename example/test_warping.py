
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from imutils import normalize_image
import imregistration as imreg
import imutils as imu

warped_data_og = imu.load_data('data/example_warped.tif', videothresh=(0, 201), fix_cut=False)
warped_data = imu.load_data('example_warped.tif', videothresh=(0, 201), fix_cut=False)

diff = warped_data_og - warped_data