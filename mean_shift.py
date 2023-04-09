import numpy as np
from utils import *

class Tracker():
    def __init__(self, num_iterations=20, kernel_type="epanechnikov", bins=16, rescale_size=None, eps=1e-3, kernel_sigma=1.5, alpha=0.0, thr=0.2):
        self.num_iterations = num_iterations
        self.kernel_type = kernel_type
        self.kernel_sigma = kernel_sigma
        self.bins = bins
        self.rescale_size = None if not rescale_size else ((rescale_size[0] // 2) * 2 + 1, (rescale_size[1] // 2) * 2 + 1)
        self.rescale_factor = (1.0, 1.0)
        self.eps = eps
        self.alpha = alpha
        self.thr = thr

    def initialize(self, image, region):
        self.image = image
        self.resized_image = image
        center, size = get_region_params(region, image.shape)
        self.center = center
        half_window = size[0] // 2, size[1] // 2
        self.size = int(half_window[0] * 2 + 1), int(half_window[1] * 2 + 1)

        if self.rescale_size:
            factorx = self.rescale_size[0] / float(size[0])
            factory = self.rescale_size[0] / float(size[1])
            self.rescale_factor = factorx, factory
            self.resized_image = cv2.resize(image, (0, 0), fx=factorx, fy=factory, interpolation=cv2.INTER_AREA) # type: ignore
            self.center = (self.center[0] * factorx, self.center[1] * factory)
            self.size = self.rescale_size
            half_window = self.size[0] // 2, self.size[1] // 2


        self.xx, self.yy = np.meshgrid(np.arange(-half_window[0], half_window[0] + 1), np.arange(-half_window[1], half_window[1] + 1))

        if self.kernel_type == "epanechnikov":
            self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.kernel_sigma)
            self.deriv_kernel = np.ones((self.size[1], self.size[0])) / (self.size[0] * self.size[1])
        else:
            self.kernel = np.ones((self.size[1], self.size[0])) / (self.size[0] * self.size[1])
            self.deriv_kernel = np.ones((self.size[1], self.size[0])) / (self.size[0] * self.size[1])

        patch = get_patch(self.resized_image, self.center, self.size)
        self.template = extract_histogram(patch[0], self.bins, weights=self.kernel * patch[1])
        self.template /= np.sum(self.template)

    def track(self, image):
        resized_image = image
        if self.rescale_size:
            resized_image = \
                cv2.resize(image, (0, 0), fx=self.rescale_factor[0], fy=self.rescale_factor[1], interpolation=cv2.INTER_AREA) # type: ignore
        
        histogram = self.template
        for _ in range(self.num_iterations):
            patch = get_patch(resized_image, self.center, self.size)
            histogram = extract_histogram(patch[0], self.bins, weights=self.kernel * patch[1])
            histogram /= np.sum(histogram)
            hist_weights = np.sqrt(self.template / (histogram + self.eps))
            backprojection = backproject_histogram(patch[0], hist_weights, self.bins)
            backprojection = backprojection * self.deriv_kernel
            x, y = self.center
            sum_backprojection = np.sum(backprojection)
            x_prev = x
            y_prev = y
            x = np.sum(backprojection * (x + self.xx)) / sum_backprojection
            y = np.sum(backprojection * (y + self.yy)) / sum_backprojection
            self.center = (x, y)

            if ((x - x_prev)**2 + (y - y_prev)**2)**(1/2) < self.thr:
                break

        self.template = (1 - self.alpha) * self.template + self.alpha * histogram
        half_window = self.size[0] // 2, self.size[1] // 2
        rerescale = (1.0 / self.rescale_factor[0], 1.0 / self.rescale_factor[1])
        left = max(round((self.center[0] - half_window[0]) * rerescale[0]), 0)
        right = min(round((self.center[0] + half_window[0]) * rerescale[0]), image.shape[1] - 1)
        top = max(round((self.center[1] - half_window[1]) * rerescale[1]), 0)
        bottom = min(round((self.center[1] + half_window[1]) * rerescale[1]), image.shape[0] - 1)
        return [left, top, right - left + 1, bottom - top + 1]