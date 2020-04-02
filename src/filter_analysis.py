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
from metrics import ClusterMetrics
import json
from time import time
from hwcounter import Timer, count, count_end


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
        answer = {'Image_index': [], 'Filter_name': [], 'count': [], 'Filter_parameter': [], 'time': [],
                  'ROC': {'array': [],
                          'energy': [],
                          'method': [],
                          'threshold': []
                          }
                  }
        for file_name in full_files:
            f = h5py.File(file_name, 'r')
            print("Start Analysis")
            obj_x_train = f['x_train']
            obj_y_train = f['y_train']
            size = obj_x_train.shape
            im_dim = int(np.sqrt(size[1]))
            n_images = size[0]
            bar = Bar('Loading', fill='@', suffix='%(percent)d%%')
            for image_index in range(0, n_images):
                a = time()
                im_real = obj_x_train[image_index, :].reshape(im_dim, im_dim)
                im_truth = obj_y_train[image_index, :].reshape(im_dim, im_dim)
                im_real = self.im_rebin(im_real, rebin_factor=4)
                im_truth = self.im_rebin(im_truth, rebin_factor=4)
                im_ped = self.im_rebin(ped, rebin_factor=4)
                im_no_pad = im_real - im_ped
                im_bin = im_truth > 0
                answer['count'].append(str(im_bin.sum()))
                denoising_filter = DenoisingFilters(im_no_pad)
                image_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])
                bar2 = Bar('Filtering image ' + str(image_index), fill='#', suffix='%(percent)d%%')
                for key in filters:
                    filter_name = key + '_filter'
                    func = getattr(DenoisingFilters, filter_name)
                    params = filters[key]
                    for param in params:
                        if param is 'lut':
                            param = [image_index]
                        with Timer() as t1:
                            image_filtered = func(denoising_filter, *param)
                        answer['time'].append(str(t1.cycles))
                        image_filtered_standardized = (image_filtered-image_filtered.mean())/image_filtered.std()
                        image_batch = np.append(image_batch, image_filtered_standardized.reshape((1,)+image_filtered.shape), axis=0)
                        answer['Filter_name'].append(key)
                        answer['Filter_parameter'].append(param)
                        bar2.next()
                bar2.finish()
                metrics = Metrics(im_no_pad, image_batch, im_bin, std)
                for threshold_method in ['local', 'global']:
                    roc, energy, threshold_array = metrics.roc_build(method=threshold_method)
                    answer['ROC']['array'].append(roc)
                    answer['ROC']['energy'].append(energy)
                    answer['ROC']['method'].append(threshold_method)
                    answer['ROC']['threshold'].append(threshold_array)
                answer['Image_index'].append(image_index)
                bar.next()
                b = time()-a
                remaining = (n_images-image_index)*b
                print('\n Estimated time per image = ' + str(b))
                print('\n Remaining (apx) = ' + str(round(remaining/60))+' minutes')

            bar.finish()

        dumped = json.dumps(answer, cls=NumpyEncoder)
        with open(path+'.json', 'w') as f:
            json.dump(dumped, f)

    def cluster_calc(self, filters, threshold):

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
                image_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])
                bar2 = Bar('Filtering image ' + str(image_index), fill='#', suffix='%(percent)d%%')
                for key in filters:
                    filter_name = key + '_filter'
                    func = getattr(DenoisingFilters, filter_name)
                    params = filters[key]
                    for param in params:
                        if param is 'lut':
                            param = [image_index]
                        with Timer() as t1:
                            image_filtered = func(denoising_filter, *param)
                            if ((key == 'mean') and (param[0] == 1)):
                                key_none = 'none'
                                best_im_bin = image_filtered > threshold[key_none]
                            else:
                                best_im_bin = image_filtered > threshold[key]
                            cluster_metrics = ClusterMetrics(best_im_bin, im_bin, im_no_pad)
                            table_truth, table_real, cluster_truth, cluster_real = cluster_metrics.get_clusters_features()
                            df_truth = pd.DataFrame(table_truth)
                            df_real = pd.DataFrame(table_real)
                        #image_filtered_standardized = (image_filtered - image_filtered.mean()) / image_filtered.std()
                        #image_batch = np.append(image_batch,
                        #                        image_filtered_standardized.reshape((1,) + image_filtered.shape),
                        #                        axis=0)
                        bar2.next()
                #bar2.finish()
                #threshold_array = []
                #for t in threshold:
                #    threshold_array.append(threshold[t])
                #threshold_array = np.array(threshold_array).reshape(-1, 1, 1)
                #std_threshold = threshold_array*std
                #best_im_bin = image_batch > std_threshold
                #metrics = Metrics(im_no_pad, image_batch, im_bin, std)
                bar.next()
                b = time() - a
                remaining = (n_images - image_index) * b
                print('\n Estimated time per image = ' + str(b))
                print('\n Remaining (apx) = ' + str(round(remaining / 60)) + ' minutes')
            bar.finish()


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
    #data.calc_metrics(filters=filters, path=path)
    data.cluster_calc(filters=filter_settings.best_filters, threshold=filter_settings.threshold_parameters)


if __name__ == "__main__":
    main()
