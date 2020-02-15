from skimage import filters
from skimage import restoration
from skimage.morphology import square
import numpy as np
from settings import FilterSettings
import cv2





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
        selem = square(window_size)
        image_norm, delta, low = self.standardize()
        filtered_image = filters.rank.mean(image_norm, selem=selem)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

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


