import numpy as np
import cv2

import random as rnd
import math

from utils import *

ALPHA = 0.1
SIGMA = 2.0
LAMBDA = 1e-8

SCALE_DIFF = 0.1

class Tracker:
    def __init__(self, alpha: float, sigma: float, lmd: float, rescale: float = 1.5, scale_diff: float = 0.0, mosse: bool = False):
        self.alpha = alpha
        self.sigma = sigma
        self.lmd = lmd
        self.scale_diff = scale_diff
        self.scale = 1.0
        self.mosse = mosse
        self.gauss: np.ndarray
        self.hanning: np.ndarray
        self.A: np.ndarray
        self.B: np.ndarray
        self.H_f_c: np.ndarray
        self.center: tuple[float, float]
        self.original_size: tuple[float, float]
        self.size: tuple[float, float]
        self.rescale = rescale
        self.initialized = False
    
    def update(self, patch):
        g_f = np.fft.fft2(self.gauss, axes=(0, 1))
        f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
        f_f_c = np.conj(f_f)
        h_f_c_n = g_f * f_f_c
        h_f_c_d = f_f * f_f_c

        if self.mosse:
            self.A = self.alpha * h_f_c_n + (1 - self.alpha) * self.A
            self.B = self.alpha * h_f_c_d + (1 - self.alpha) * self.B
            self.H_f_c = self.A / (self.B + self.lmd)
        else:
            self.H_f_c = self.alpha * h_f_c_n / (h_f_c_d + self.lmd) + (1 - self.alpha) * self.H_f_c

    def create_gauss_ch(self, channels) -> np.ndarray:
        peak = create_gauss_peak(self.size, self.sigma)
        if channels == 1:
            return peak
        else:
            return np.repeat(np.expand_dims(peak, 2), channels, axis=2)
        
    def create_hanning_ch(self, channels) -> np.ndarray:
        peak = create_cosine_window(self.size)
        if channels == 1:
            return peak
        else:
            return np.repeat(np.expand_dims(peak, 2), channels, axis=2)

    def initialize_mosse(self, img: np.ndarray, center: tuple[float, float], size: tuple[float, float], num_transforms: int = 8):
        patch, _ = get_patch(img, center, self.size)
        g_f = np.fft.fft2(self.gauss, axes=(0, 1))
        f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
        f_f_c = np.conj(f_f)
        h_f_c_n = g_f * f_f_c
        h_f_c_d = f_f * f_f_c
        self.A = h_f_c_n
        self.B = h_f_c_d

        for i in range(num_transforms):
            tx = rnd.randint(int(-size[0] // 8), int(size[0] // 8))
            ty = rnd.randint(int(-size[1] // 8), int(size[1] // 8))
            angle = rnd.random() * 2 * np.pi
            scale = 1.0 + rnd.random() * 0.4 - 0.2
            shearx = rnd.random() - 0.5
            sheary = rnd.random() - 0.5
            E = np.array([[math.cos(angle), -math.sin(angle), tx], [math.sin(angle), math.cos(angle), ty]]) # type: ignore
            S = np.array([[scale, shearx, 0], [scale, sheary, 0]]) # type: ignore
            img_t = cv2.warpAffine(img, E, (img.shape[1], img.shape[0]))
            img_t = cv2.warpAffine(img_t, S, (img.shape[1], img.shape[0]))
            patch, _ = get_patch(img, center, self.size)
            g_f = np.fft.fft2(self.gauss, axes=(0, 1))
            f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
            f_f_c = np.conj(f_f)
            h_f_c_n = g_f * f_f_c
            h_f_c_d = f_f * f_f_c
            self.A += h_f_c_n
            self.B += h_f_c_d
        
        self.A *= 1.0 / (num_transforms + 1)
        self.B *= 1.0 / (num_transforms + 1)

        self.H_f_c = self.A / (self.B + self.lmd)

    def initialize(self, img: np.ndarray, region: np.ndarray):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.initialized = True
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        center, size = get_region_params2(region)

        self.center = center
        self.original_size = size
        self.size = (make_odd(round(size[0] * self.rescale)), make_odd(round(size[1] * self.rescale)))
        self.gauss = self.create_gauss_ch(img.shape[2] if len(img.shape) > 2 else 1)
        self.hanning = self.create_hanning_ch(img.shape[2] if len(img.shape) > 2 else 1)

        if self.mosse:
            self.initialize_mosse(img, center, self.size)
        else:
            patch, _ = get_patch(img, center, self.size)
            g_f = np.fft.fft2(self.gauss, axes=(0, 1))
            f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
            f_f_c = np.conj(f_f)
            h_f_c_n = g_f * f_f_c
            h_f_c_d = f_f * f_f_c
            self.H_f_c = h_f_c_n / (h_f_c_d + self.lmd)

    def track_std(self, img: np.ndarray):
        patch, _ = get_patch(img, self.center, self.size)
        f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
        h = np.fft.ifft2(f_f * self.H_f_c, axes=(0, 1))
        if (len(img.shape) > 2):
            h = np.linalg.norm(h, 2, axis=2)
        max_idx: tuple[int, int] = np.unravel_index(np.argmax(h), h.shape) # type: ignore
        self.center = update_center(self.center, max_idx, self.size, 1.0, img)

        patch, _ = get_patch(img, self.center, self.size)
        self.update(patch)

        return self.center, self.original_size

    def track_resize(self, img: np.ndarray):
        scale_diff = self.scale * self.scale_diff
        img_small = cv2.resize(img, (0, 0), fx=self.scale - scale_diff, fy=self.scale - scale_diff) # type: ignore
        img_large = cv2.resize(img, (0, 0), fx=self.scale + scale_diff, fy=self.scale + scale_diff) # type: ignore
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale) # type: ignore
        patch, _ = get_patch(img, (self.center[0] * self.scale, self.center[1] * self.scale), self.size)
        patch_small, _ = get_patch(img_small, (self.center[0] * (self.scale - scale_diff), self.center[1] * (self.scale - scale_diff)), self.size)
        patch_large, _ = get_patch(img_large, (self.center[0] * (self.scale + scale_diff), self.center[1] * (self.scale + scale_diff)), self.size)

        f_f = np.fft.fft2(patch * self.hanning, axes=(0, 1))
        f_f_small = np.fft.fft2(patch_small * self.hanning, axes=(0, 1))
        f_f_large = np.fft.fft2(patch_large * self.hanning, axes=(0, 1))
        h = np.linalg.norm(np.fft.ifft2(f_f * self.H_f_c, axes=(0, 1)), 2, axis=2)
        h_small = np.linalg.norm(np.fft.ifft2(f_f_small * self.H_f_c, axes=(0, 1)), 2, axis=2)
        h_large = np.linalg.norm(np.fft.ifft2(f_f_large * self.H_f_c, axes=(0, 1)), 2, axis=2)

        max_i = np.argmax([np.max(h), np.max(h_small), np.max(h_large)])
        if max_i == 0:
            max_idx: tuple[int, int] = np.unravel_index(np.argmax(h), h.shape) # type: ignore
            self.center = update_center(self.center, max_idx, self.size, self.scale, img)
        elif max_i == 1:
            max_idx: tuple[int, int] = np.unravel_index(np.argmax(h_small), h_small.shape) # type: ignore
            self.center = update_center(self.center, max_idx, self.size, self.scale - scale_diff, img)
            self.scale = self.scale - scale_diff
        else:
            max_idx: tuple[int, int] = np.unravel_index(np.argmax(h_large), h_large.shape) # type: ignore
            self.center = update_center(self.center, max_idx, self.size, self.scale + scale_diff, img)
            self.scale = self.scale + scale_diff

        patch, _ = get_patch(img, self.center, self.size)
        self.update(patch)

        return self.center, (self.original_size[0] / self.scale, self.original_size[1] / self.scale)

    def track(self, img: np.ndarray):
        if not self.initialized:
            raise Exception("Tracker not initialized")
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        if self.scale_diff == 0.0:
            center, size = self.track_std(img)
        else:
            center, size = self.track_resize(img)
        half_window = size[0] / 2.0, size[1] / 2.0
        left = max(round(center[0] - half_window[0]), 0)
        top = max(round(center[1] - half_window[1]), 0)
        return [left, top, self.size[0], self.size[1]]
        