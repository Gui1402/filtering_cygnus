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
from sklearn.preprocessing import MinMaxScaler


def image_rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def arrrebin(img, rebin):
    newshape = int(2048 / rebin)
    img_rebin = image_rebin(img, (newshape, newshape))
    return img_rebin


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
        self.answer = {'Image_index': [], 'Filter_name': [], 'count': [], 'Filter_parameter': [], 'time': [],
                       'ROC': {'array': [],
                               'energy': [],
                               'energy_real': [],
                               'energy_sdv': [],
                               'method': [],
                               'threshold': []
                              }
                      }

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
            im_rebined = im_input[yy, xx]
        return im_rebined

    # get pedestal from noise file
    def get_pedestal(self):
        """ load pedestal file that has mean and std images
            input None
            output ped and std image map"""
        fn = h5py.File(self.file_noise, 'r')  # read file noise
        ped = fn['mean']  # get pedestal
        std = fn['std']  # get std
        index = abs(817-self.run_number)
        return ped[:, :, index], std[:, :, index]  # return values according to run number

    # get all h5 simulated files
    def get_file_names(self):
        """ get all .h5 files in folder
            input None
            output all h5 files in a list"""
        return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder

    def get_filter_results(self, im_no_pad, image_batch, im_bin, std, im_truth, keys):
        if 'cygno' in keys:
            cygno_index = np.where(keys == 'cygno')[0]
            image_cygno_batch = []
            metrics_cygno_object = Metrics(im_no_pad, image_cygno_batch, im_bin, std, im_truth)
            roc_cy, energy_cy, energy_real_cy, energy_sdv_cy, threshold_array_cy = metrics_cygno_object.roc_build(method='global')
            image_batch = np.delete(image_batch, cygno_index, axis=0)
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
            roc, energy, energy_real, energy_sdv, threshold_array = metrics.roc_build(method='global')
            roc0 = np.append(roc_cy[0], roc[0], axis=1)
            roc1 = np.append(roc_cy[1], roc[1], axis=1)
            roc = (roc0, roc1)
            energy = np.append(energy_cy, energy, axis=1)
            energy_real = np.append(energy_real_cy, energy_real, axis=1)
            energy_sdv = np.append(energy_sdv_cy, energy_sdv, axis=1)
            threshold_array = np.append(threshold_array_cy.reshape(-1, 1, 1, 1), threshold_array, axis=1)
        else:
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
            roc, energy, energy_real, energy_sdv, threshold_array = metrics.roc_build(method='global')

        scores = metrics.roc_score(roc, threshold_array, param=0.90)
        self.answer['ROC']['array'].append(roc)
        self.answer['ROC']['energy'].append(energy)
        self.answer['ROC']['energy_real'].append(energy_real)
        self.answer['ROC']['energy_sdv'].append(energy_sdv)
        self.answer['ROC']['threshold'].append(threshold_array)
        return scores

    def cluster_preprocess(self, image_batch_input, scores, im_no_pad, im_bin, std, keys, mode):
        mode_key = mode + '_constant'
        scores = np.array(scores[mode_key][0])
        threshold_array = scores[1, :]
        threshold_matrix = np.append(std.reshape((1,) + im_bin.shape), np.ones(shape=((len(keys)-1,)+im_bin.shape)),
                                     axis=0)

        best_images = image_batch_input > (threshold_array.reshape(-1, 1, 1))*threshold_matrix
        sg_amount = im_bin.sum()
        count_bg_before = []
        count_bg_after = []
        for image_index in range(best_images.shape[0]):
            # check amount of bg_pixels
            xbg, ybg = np.where(im_bin == False)
            best_image = best_images[image_index, ...]
            h, w = best_image.shape
            bg_amount = best_image[xbg, ybg].sum() / ((h * w) - sg_amount)
            #print("Before rebin :", 100 * bg_amount)
            img_rb_zs = arrrebin(best_image, rebin=4) > 0
            img_rb_truth = arrrebin(im_bin, rebin=4) > 0

            sg_rebin_amount = (img_rb_truth > 0).sum()
            x_rebin_bg, y_rebin_bg = np.where(img_rb_truth == False)
            h_rebin, w_rebin = img_rb_truth.shape
            bg_rebin_amount = img_rb_zs[x_rebin_bg, y_rebin_bg].sum() / ((h_rebin * w_rebin) - sg_rebin_amount)
            count_bg_before.append(bg_amount)
            count_bg_after.append(bg_rebin_amount)

        return count_bg_before, count_bg_after

    def get_clusters(self, image_batch_input, scores, im_no_pad, im_bin, mode):
        mode_key = mode + '_constant'
        scores = np.array(scores[mode_key][0])
        threshold_array = scores[1, :]
        best_images = image_batch_input > threshold_array.reshape(-1, 1, 1)
        cluster_return = {'cluster': [],
                          'info': []
                          }
        truth_return = {'cluster': [],
                        'info': []
                        }

        for image_index in range(best_images.shape[0]):
            best_image = best_images[image_index, ...]
            cluster_metrics = ClusterMetrics(best_image, im_bin, im_no_pad)
            table_truth, table_real, cluster_truth, cluster_real = cluster_metrics.get_clusters_features()
            df_truth = pd.DataFrame(table_truth)
            df_truth['truth'] = True
            df_real = pd.DataFrame(table_real)
            df_real['truth'] = False
            df_merged = df_truth.append(df_real)
            scaler = MinMaxScaler()
            scaled_array = scaler.fit_transform(df_merged.drop('truth', axis=1))
            correlation_matrix = np.corrcoef(scaled_array)
            got_cluster_index = (-correlation_matrix[0, :]).argsort()[1] - 1
            cluster_return['cluster'].append(cluster_real[got_cluster_index])
            cluster_return['info'].append(df_real.iloc[got_cluster_index, :])
        truth_return['cluster'].append(cluster_truth[0])
        truth_return['info'].append(df_truth.iloc[0, :])
        return cluster_return, truth_return

    def calc_metrics(self, filters, path):
        """Apply filter on images and calculate metrics
           input filters that will be applied and folder where the output file will be saved
           output an output file with results"""
        full_files = self.get_file_names()
        im_ped, std = self.get_pedestal()
        #std = self.im_rebin(std, rebin_factor=4)
        before_mean = 0
        after_mean = 0
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
                #im_real = self.im_rebin(im_real, rebin_factor=4)
                #im_truth = self.im_rebin(im_truth, rebin_factor=4)
                #im_ped = self.im_rebin(ped, rebin_factor=4)
                im_no_pad = im_real - im_ped
                im_bin = im_truth > 0
                self.answer['count'].append(str(im_bin.sum()))
                denoising_filter = DenoisingFilters(im_no_pad)
                image_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])
                for key in filters:
                    if key is not 'cygno':
                        filter_name = key + '_filter'
                        func = getattr(DenoisingFilters, filter_name)
                        params = filters[key]
                        for param in params:
                            if param is 'lut':
                                param = [image_index]
                            with Timer() as t1:
                                image_filtered = func(denoising_filter, *param)
                            self.answer['time'].append(str(t1.cycles))
                            image_filtered_standardized = (image_filtered-image_filtered.mean())/image_filtered.std()
                            image_batch = np.append(image_batch, image_filtered_standardized.reshape((1, ) +
                                                    image_filtered.shape), axis=0)
                            self.answer['Filter_name'].append(key)
                            self.answer['Filter_parameter'].append(param)
                    else:
                        image_batch = np.append(image_batch, im_no_pad.reshape((1,) + im_no_pad.shape), axis=0)
                        self.answer['Filter_name'].append(key)
                        self.answer['Filter_parameter'].append('none')
                        self.answer['time'].append(str(0))

                keys_array = np.array(list(filters.keys()))
                best_results = self.get_filter_results(im_no_pad, image_batch, im_bin, std, im_truth, keys_array)
                before, after = self.cluster_preprocess(image_batch, best_results, im_no_pad, im_bin, std, keys_array, mode='bg')
                before_mean += np.array(before).mean()
                after_mean += np.array(after).mean()
                print('Before ', before_mean/(image_index+1))
                print('\n After ', after_mean/(image_index+1))

                #print(result)
                self.answer['Image_index'].append(image_index)
                bar.next()
                b = time()-a
                remaining = (n_images-image_index)*b
                print('\n Estimated time per image = ' + str(b))
                print('\n Remaining (apx) = ' + str(round(remaining/60))+' minutes \n')

            bar.finish()

        dumped = json.dumps(self.answer, cls=NumpyEncoder)
        with open(path+'.json', 'w') as f:
            json.dump(dumped, f)

def main():
    mode = "cluster"
    filter_settings = FilterSettings()
    data_folder = filter_settings.data_folder
    noise_file = filter_settings.noise_file
    run_number = filter_settings.run_number
    sup = filter_settings.sup
    inf = filter_settings.inf
    roc_grid = filter_settings.roc_grid
    data = ResultGeneration(data_folder, noise_file, run_number, sup, inf, roc_grid)
    filters = filter_settings.best_filters
    path = filter_settings.output_file_path + filter_settings.output_file_name
    data.calc_metrics(filters=filters, path=path)
    #data.cluster_calc(filters=filter_settings.best_filters, threshold=filter_settings.threshold_parameters)


if __name__ == "__main__":
    main()

