import numpy as np
import pandas as pd
import h5py
import glob
import itertools
from progress.bar import Bar
from time import time
from settings import FilterSettings
from filters import DenoisingFilters
from metrics import Metrics
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ResultGeneration:

    def __init__(self, folder, file_noise, run_number, bound_sup, bound_inf, roc_grid):
        self.folder = folder  # folder where there are h5 files
        self.file_noise = file_noise  # folder that has the noise files
        self.run_number = run_number  # the run number for the noise
        self.bound_sup = bound_sup    # the upper bound to remove outliers
        self.bound_inf = bound_inf    # the lower bound to remove outliers
        self.roc_grid = roc_grid      # number of points to generate roc curve

    def im_rebin(self, im_input, rebin_factor=8):
        """ rebin an input image by a rebin factor
            input (N,M,k)
            output (N//factor, M//factor, k)"""
        dim = im_input.shape
        index = range(0, dim[0], rebin_factor)
        try:
            n_imgs = dim[2]
            xx, yy, zz = np.meshgrid(index, index, range(0, n_imgs))
            im_rebined = im_input[xx, yy, zz]
        except IndexError:
            n_imgs = 1
            xx, yy = np.meshgrid(index, index)
            im_rebined = im_input[xx, yy]
        return im_rebined

    # get pedestal from noise file
    def get_pedestal(self):
        """ load pedestal file that has mean and std images
            input None
            output ped and std image map"""
        fn = h5py.File(self.file_noise, 'r')  # read file noise
        ped = fn['mean']  # get pedestal
        std = fn['std']  # get std
        return ped[:, :, 820-self.run_number], std[:, :, 820-self.run_number]  # return values according to run number

    # get all h5 simulated files
    def get_file_names(self):
        """ get all .h5 files in folder
            input None
            output all h5 files in a list"""
        return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder

    def calc_metrics(self, filters, path):
        """Apply filter on images and calculate metrics
           input filters that will be applied and folder where the output file will be saved
           output an output file with results"""
        full_files = self.get_file_names()
        ped, std = self.get_pedestal()
        std = self.im_rebin(std, rebin_factor=4)
        for file_name in full_files:
            f = h5py.File(file_name, 'r')
            print("Start Analysis")
            obj_x_train = f['x_train']
            obj_y_train = f['y_train']
            size = obj_x_train.shape
            im_dim = int(np.sqrt(size[1]))
            n_images = size[0]
            bar = Bar('Loading', fill='@', suffix='%(percent)d%%')
            answer = {'Image_index': [], 'Filter_name': [], 'Filter_parameter': [], 'ROC': [], 'AUC': []}
            for image_index in range(0, n_images):
                a = time()
                im_real = obj_x_train[image_index, :].reshape(im_dim, im_dim)
                im_truth = obj_y_train[image_index, :].reshape(im_dim, im_dim)
                im_real = self.im_rebin(im_real, rebin_factor=4)
                im_truth = self.im_rebin(im_truth, rebin_factor=4)
                im_ped = self.im_rebin(ped, rebin_factor=4)
                im_no_pad = im_real - im_ped
                im_bin = im_truth > 0
                denoising_filter = DenoisingFilters(im_no_pad)
                for key in filters:
                    filter_name = key + '_filter'
                    func = getattr(DenoisingFilters, filter_name)
                    params = filters[key]
                    for param in params:
                        if param is 'lut':
                            param = image_index
                        image_filtered = func(denoising_filter, *param)
                        metrics = Metrics(im_no_pad, image_filtered, im_bin, std)
                        roc = metrics.roc_build()[0, :, :]
                        auc = metrics.calc_auc(roc)
                        answer['Image_index'].append(image_index)
                        answer['Filter_name'].append(key)
                        answer['Filter_parameter'].append(param)
                        answer['ROC'].append(roc)
                        answer['AUC'].append(auc)
                bar.next()
                b = time()-a
                remaining = (n_images-image_index)*b
                print('\n Estimated time per image = ' + str(b))
                print('\n Remaining (apx) = ' + str(round(remaining/60))+' minutes')

            bar.finish()
        dumped = json.dumps(answer, cls=NumpyEncoder)
        with open(path+'.json', 'w') as f:
            json.dump(dumped, f)


def main():
    filter_settings = FilterSettings()
    data_folder = filter_settings.data_folder
    noise_file = filter_settings.noise_file
    run_number = filter_settings.run_number
    sup = filter_settings.sup
    inf = filter_settings.inf
    roc_grid = filter_settings.roc_grid
    data = ResultGeneration(data_folder, noise_file, run_number, sup, inf, roc_grid)
    filters = filter_settings.filters
    path = filter_settings.output_file_path + filter_settings.output_file_name
    data.calc_metrics(filters=filters, path=path)


if __name__ == "__main__":
    main()
