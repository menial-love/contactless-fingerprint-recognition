import numpy as np
import cv2
from ridge import ridge_segment, ridge_orient, ridge_freq, ridge_filter


# 标准伽马函数校正RGB图像
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


# rgb转灰度图
def rgb2gray(im: np.ndarray, rgb2gray_vector: np.ndarray) -> np.ndarray:
    # [0.2989, 0.5870, 0.1140]
    # rgb2gray_vector = np.asarray([0.2126, 0.7152, 0.0722]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        img_gray = np.copy(im)
    elif im.shape[2] == 1:
        img_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        img_gray = np.dot(im, rgb2gray_vector)
        img_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return img_gray.astype(np.float32)


def image_enhance(image):
    block_size = 16   # 8
    threshold = 0.1   # 0.01
    normalized_image, mask = ridge_segment(image, block_size, threshold)  # Normalize the image and find a ROI

    gradient_sigma = 1
    block_sigma = 7
    orientation_smooth_sigma = 7
    orientation_image = ridge_orient(normalized_image, gradient_sigma, block_sigma, orientation_smooth_sigma)  # Find orientation of every pixel

    block_size = 38   # 10
    window_size = 5   # 3
    min_wavelength = 5  # 3
    max_wavelength = 15  # 10
    frequency, median_frequency = ridge_freq(normalized_image, mask, orientation_image, block_size, window_size,
                                             min_wavelength, max_wavelength)  # Find the overall frequency of ridges

    frequency = median_frequency * mask
    kx = 0.65   # 0.5
    ky = 0.65   # 0.5
    # Create gabor filter and do the actual filtering
    new_image = ridge_filter(normalized_image, orientation_image, frequency, kx, ky)

    return new_image > 1
