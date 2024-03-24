import numpy as np
import scipy.ndimage
import math


def freQuest(image, orientation_image, window_size, min_wave_length, max_wave_length):
    rows, cols = np.shape(image)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.
    cos_orientation = np.mean(np.cos(2 * orientation_image))
    sin_orientation = np.mean(np.sin(2 * orientation_image))
    orientation = math.atan2(sin_orientation, cos_orientation) / 2

    # Rotate the image block so that the ridges are vertical
    rotated_image = scipy.ndimage.rotate(image, orientation / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                         mode='nearest')

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.
    crop_size = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - crop_size) / 2))
    cropped_image = rotated_image[offset:offset + crop_size][:, offset:offset + crop_size]

    # Sum down the columns to get a projection of the grey values down
    # the ridges.
    projection = np.sum(cropped_image, axis=0)
    dilation = scipy.ndimage.grey_dilation(projection, window_size, structure=np.ones(window_size))

    temp = np.abs(dilation - projection)

    peak_threshold = 2

    max_points = (temp < peak_threshold) & (projection > np.mean(projection))
    max_indices = np.where(max_points)

    rows_max_indices, cols_max_indices = np.shape(max_indices)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0
    if cols_max_indices < 2:
        frequency_image = np.zeros(image.shape)
    else:
        no_of_peaks = cols_max_indices
        wavelength = (max_indices[0][cols_max_indices - 1] - max_indices[0][0]) / (no_of_peaks - 1)
        if min_wave_length <= wavelength <= max_wave_length:
            frequency_image = 1 / np.double(wavelength) * np.ones(image.shape)
        else:
            frequency_image = np.zeros(image.shape)

    return frequency_image
