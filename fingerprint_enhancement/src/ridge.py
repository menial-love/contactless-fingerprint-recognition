import numpy as np
import cv2
from scipy import signal, ndimage
from frequest import freQuest


def normalise(img):
    normed = (img - np.mean(img)) / np.std(img)
    return normed


def ridge_segment(image, block_size, threshold):
    rows, cols = image.shape

    image = cv2.equalizeHist(image)
    new_rows = np.int(block_size * np.ceil((np.float(rows)) / (np.float(block_size))))
    new_cols = np.int(block_size * np.ceil((np.float(cols)) / (np.float(block_size))))

    padded_image = np.zeros((new_rows, new_cols))
    std_dev_image = np.zeros((new_rows, new_cols))

    padded_image[0:rows][:, 0:cols] = image

    for i in range(0, new_rows, block_size):
        for j in range(0, new_cols, block_size):
            block = padded_image[i:i + block_size][:, j:j + block_size]

            std_dev_image[i:i + block_size][:, j:j + block_size] = np.std(block) * np.ones(block.shape)

    std_dev_image = std_dev_image[0:rows][:, 0:cols]
    mask = std_dev_image > threshold
    mean_value = np.mean(image[mask])
    std_value = np.std(image[mask])
    normalized_image = (image - mean_value) / std_value

    return normalized_image, mask


def ridge_orient(image, gradient_sigma, block_sigma, orient_smooth_sigma):
    # Calculate image gradients
    sze = np.fix(6 * gradient_sigma)
    if np.remainder(sze, 2) == 0:
        sze = sze + 1

    gauss = cv2.getGaussianKernel(np.int(sze), gradient_sigma)
    f = gauss * gauss.T

    fy, fx = np.gradient(f)  # Gradient of Gaussian

    Gx = signal.convolve2d(image, fx, mode='same')
    Gy = signal.convolve2d(image, fy, mode='same')

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    # Now smooth the covariance data to perform a weighted summation of the data
    sze = np.fix(6 * block_sigma)

    gauss = cv2.getGaussianKernel(np.int(sze), block_sigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

    sin2theta = Gxy / denom  # Sine and cosine of doubled angles
    cos2theta = (Gxx - Gyy) / denom

    if orient_smooth_sigma:
        sze = np.fix(6 * orient_smooth_sigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1
        gauss = cv2.getGaussianKernel(np.int(sze), orient_smooth_sigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta, f)    # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta, f)    # doubled angles

    orient_image = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2
    return orient_image


def ridge_freq(image, mask, orientation, block_size, window_size, min_wavelength, max_wavelength):
    rows, cols = image.shape
    frequencies = np.zeros((rows, cols))

    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            block_image = image[r:r + block_size][:, c:c + block_size]
            block_orientation = orientation[r:r + block_size][:, c:c + block_size]

            frequencies[r:r + block_size][:, c:c + block_size] = freQuest(block_image, block_orientation, window_size,
                                                                          min_wavelength, max_wavelength)

    frequencies = frequencies * mask
    frequencies_1d = np.reshape(frequencies, (1, rows * cols))
    ind = np.where(frequencies_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = frequencies_1d[0][ind]

    mean_frequency = np.mean(non_zero_elems_in_freq)
    return frequencies, mean_frequency


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.

    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.int(np.round(3 * np.max([sigmax, sigmay])))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay))) * np.cos(
        2 * np.pi * unfreq[0] * x)  # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    angleRange = np.int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.
        rot_filt = ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

    # do the filtering

    for i in range(0, rows):
        for j in range(0, cols):
            if (orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if (orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim
