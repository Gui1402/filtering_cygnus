from skimage import filters
from skimage import restoration
from skimage.morphology import square
import numpy as np
import pybind_bm3d as m
from settings import FilterSettings
import h5py
from scipy import ndimage
import sys
from dn_cnn_test import NnTest
sys.path.append('fcdnn/')
sys.path.append('/fcdnn')
from core.test_blind_ft import Fine_tuning as test_ft
from skimage.restoration import denoise_tv_bregman

class DenoisingFilters:

    def __init__(self, image_input):
        self._image_input = image_input

    def standardize(self):
        filter_settings = FilterSettings()
        up = filter_settings.sup
        low = filter_settings.inf
        satured_image = self._image_input
        satured_image[satured_image >= up] = up
        satured_image[satured_image <= low] = low
        delta = up - low
        image_norm = (satured_image - low) / delta
        return image_norm, delta, low

    def gaussian_filter(self, window_size):
        sigma = window_size/3
        return filters.gaussian(self._image_input, sigma=sigma)

    def mean_filter(self, window_size):
        mask = np.ones((window_size, window_size))/(window_size**2)
        return ndimage.filters.convolve(self._image_input, mask)

    def median_filter(self, window_size):
        selem = square(window_size)
        return filters.median(self._image_input, selem=selem)

    def wiener_filter(self, const_psf):
        psf = np.ones((const_psf, const_psf)) / (const_psf * const_psf)
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.unsupervised_wiener(image_norm, psf)[0]
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def bilateral_filter(self, win_size, sigma_r, sigma_d):
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.denoise_bilateral(image_norm,
                                                       win_size=win_size,
                                                       sigma_color=sigma_r,
                                                       sigma_spatial=sigma_d)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def nlmeans_filter(self, patch_size, patch_distance):
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.denoise_nl_means(image_norm,
                                                      patch_size=patch_size,
                                                      patch_distance=patch_distance)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def bm3D_filter(self, sigma):
        filter_settings = FilterSettings()
        img_satured = 99 + np.clip(self._image_input, filter_settings.inf, filter_settings.sup)
        filtered_img = m.bm3d(img_satured, sigma) - 99
        return filtered_img

    def FCAIDE_filter(self, index):
        #t_ft = test_ft(self._image_input, self._image_input, sigma)
        #return t_ft.fine_tuning()

        obj = h5py.File('../data/FC-AIDE/merged.h5', 'r')
        data = obj['result']
        return data[index, :].reshape(512, 512)

    def DnCNN_filter(self, path):
        image_norm, delta, low = self.standardize()
        nn_test = NnTest(image_norm, path)
        filtered_image = nn_test.test()
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def tv_filter(self, w):
        return denoise_tv_bregman(self._image_input, w)

    def wavelets_filter(self, param):
        return param




