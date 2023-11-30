import cv2 as cv
from frame_splitter import frameSplitter
import scipy
from scipy.signal import filtfilt, butter
import numpy as np
import os
import re
import shutil
from tqdm import tqdm
from pystackreg import StackReg
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
  
def calculate_window(fps, nseconds=2):
    window = fps*nseconds
    return window

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def gaussian_filter(data,fps):
    sigma = calculate_window(fps,nseconds=2)
    return scipy.ndimage.filters.gaussian_filter(data, sigma,axes=0)

def no_filter(data,fps):
    return data

def stabilize_frames( raw_centres, fps):
    raw_centres_arr = np.array(raw_centres)
    
    # Gaussian filter
    smoothed_centres_gaussian =gaussian_filter(raw_centres_arr.squeeze(), fps)
    
    # StackReg stabilization
    sr = StackReg(StackReg.TRANSLATION)
    stabilized_centres = np.zeros_like(smoothed_centres_gaussian)
    for i in range(smoothed_centres_gaussian.shape[0]):
        transformed = sr.register_stack(smoothed_centres_gaussian[i, :, :].squeeze(), reference=smoothed_centres_gaussian[0, :, :].squeeze())
        stabilized_centres[i, :, :] = transformed

    return stabilized_centres





with open('raw_centres.npy', 'rb') as f:
    raw_signal = np.load(f)
    

# Set up the plot
plt.figure(figsize=(10, 6))

# Iterate over the second dimension (24 points)
for j in range(raw_signal.shape[1]):
    # Extract x and y coordinates for each point
    x_values = np.arange(100)  # Assuming 100 time points
    x_coordinates = raw_signal[:, j, 0]  # X coordinates for the j-th point
    y_coordinates = raw_signal[:, j, 1]  # Y coordinates for the j-th point

    # Calculate the Euclidean distance from the initial location
    initial_location = np.array([x_coordinates[0], y_coordinates[0]])
    euclidean_distances = np.sqrt((x_coordinates - initial_location[0])**2 + (y_coordinates - initial_location[1])**2)

    # Plot the total Euclidean distance for each point
    plt.plot(x_values, euclidean_distances, label=f'Point {j + 1}')

# Add labels and title
plt.xlabel('Time Points')
plt.ylabel('Euclidean Distance from Initial Location')
plt.title('Total Euclidean Distance of Each Point Over Time')
plt.legend()  # Add legend for different points

# Show the plot
plt.show()
