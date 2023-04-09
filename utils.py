import math

import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb


def gaussderiv(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    
    D = -2 * (x * np.exp(-x**2 / (2 * sigma**2))) / (np.sqrt(2 * math.pi) * sigma**3)
    D = D / (np.sum(np.abs(D)) / 2)
    
    Dx = cv2.sepFilter2D(img, -1, D, G)
    Dy = cv2.sepFilter2D(img, -1, G, D)

    return Dx, Dy

def gausssmooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)
    
def show_flow(U, V, ax, type='field', set_aspect=False):
    if type == 'field':
        scaling = 0.1
        u = cv2.resize(gausssmooth(U, 1.5), (0, 0), fx=scaling, fy=scaling) # type: ignore
        v = cv2.resize(gausssmooth(V, 1.5), (0, 0), fx=scaling, fy=scaling) # type: ignore
        
        x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
        y_ = -(np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
        x, y = np.meshgrid(x_, y_)
        
        ax.quiver(x, y, -u * 5, v * 5) # type: ignore
        if set_aspect:
            ax.set_aspect(1.)
    elif type == 'magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        ax.imshow(np.minimum(1, magnitude))
    elif type == 'angle':
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    elif type == 'angle_magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.minimum(1, magnitude), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    else:
        print('Error: unknown optical flow visualization type.')
        exit(-1)

def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def get_patch(img, center, sz):
    # crop coordinates
    x0 = round(int(center[0] - sz[0] / 2))
    y0 = round(int(center[1] - sz[1] / 2))
    x1 = int(round(x0 + sz[0]))
    y1 = int(round(y0 + sz[1]))
    # padding
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target
    if len(img.shape) > 2:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)
    return im_crop_padded, crop_mask

def create_epanechnik_kernel(width, height, sigma):
    # make sure that width and height are odd
    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))
    kernel = kernel / np.max(kernel)
    kernel[kernel<0] = 0
    return kernel

def extract_histogram(patch, nbins, weights=None):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    # zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements)
    histogram = np.zeros((nbins**3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram

def backproject_histogram(patch, histogram, nbins):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # use histogram us a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection

def get_region_params(region, image_shape):
    if len(region) == 8:
        x_ = np.array(region[::2])
        y_ = np.array(region[1::2])
        region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

    left = max(region[0], 0)
    top = max(region[1], 0)

    right = min(region[0] + region[2], image_shape[1] - 1)
    bottom = min(region[1] + region[3], image_shape[0] - 1)

    center = (left + right) / 2, (top + bottom) / 2
    size = right - left, bottom - top

    return center, size

def get_region_params2(region):
    if len(region) == 8:
        x_ = np.array(region[::2])
        y_ = np.array(region[1::2])
        region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

    left = region[0]
    top = region[1]

    right = region[0] + region[2]
    bottom = region[1] + region[3]

    center = (left + right) / 2.0, (top + bottom) / 2.0
    size = right - left, bottom - top

    return center, size

def create_cosine_window(target_size):
    # target size is in the format: (width, height)
    # output is a matrix of dimensions: (width, height)
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)

def update_center(center: tuple[float, float], max_idx: tuple[float, float], size: tuple[float, float], scale: float, img: np.ndarray) -> tuple[float, float]:
    dx = float(max_idx[1])
    dy = float(max_idx[0])
    if dx > size[0] / 2:
        dx -= size[0]
    if dy > size[1] / 2:
        dy -= size[1]

    dx /= scale
    dy /= scale
    center = center[0] + dx, center[1] + dy
    height = img.shape[0]
    width = img.shape[1]
    center = min(max(center[0], 0), width), min(max(center[1], 0), height)
    return center

def create_gauss_peak(target_size, sigma):
    # target size is in the format: (width, height)
    # sigma: parameter (float) of the Gaussian function
    # note that sigma should be small so that the function is in a shape of a peak
    # values that make sens are approximately from the interval: ~(0.5, 5)
    # output is a matrix of dimensions: (width, height)
    w2 = int(target_size[0] / 2)
    h2 = int(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    G = np.exp(-X**2 / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G

def make_odd(x):
    return x + 1 - (x % 2)