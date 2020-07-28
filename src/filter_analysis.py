import numpy as np
import h5py
import glob
from progress.bar import Bar
from settings import FilterSettings
from filters import DenoisingFilters
from metrics import Metrics
import json
from time import time
from hwcounter import Timer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from metrics import roc_score, image_rebin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from tqdm import tqdm
import argparse
import sys
import os
from skimage.morphology import label
from skimage.measure import regionprops, regionprops_table




def do_roc_plot(rf, keys_array, file):
    for index, key in list(enumerate(keys_array)):
        plt.plot(rf[0][:, index],rf[1][:, index], label=key)
    plt.legend()
    plt.grid()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()


def do_image_hist_plot(im_bin, im_no_pad, file):
    xs, ys = np.where(im_bin == True)
    xb, yb = np.where(im_bin == False)
    plt.hist(im_no_pad[xs, ys], log=True,
             histtype='step', color='blue',
             bins='scott', label='signal')
    plt.hist(im_no_pad[xb, yb], log=True,
             histtype='step', color='black',
             bins='scott', label='background')

    plt.legend()
    plt.title(file)

def animation_plot(image, threshold, std, sg, bg, name):
    """
    Create a git from a for loop of imshow

    :param image: Input image
    :param threshold: threshold array that will used to plot binary images
    :param std: std map that will be multiplied by threshold array
    :param sg: sg array that will be shown on title
    :param bg: bg array that will be shown on title
    :param name: filename at output
    :return: none

    """
    fig = plt.figure()
    ims = []
    for index, threshold_value in list(enumerate(threshold)):
        plt.title('sg-eff: ' + str(sg[index]) + '\n bg-eff: ' + str(bg[index]))
        im = plt.imshow(image > threshold_value*std, animated=True, cmap='gray')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=500)
    with open('../data/' + name + '.html', 'w') as f:
        print(ani.to_html5_video(), file=f)


def apply_dbscan(image_batch, im_bin, std_map, threshold_array, key_names, rebin_mode):
    """
    DBSCAN clustering find optimal parameters

    :param image_batch: Filtered images that will be evaluated (k, M, N)
    :param threshold_array: threshold select from roc curves (k,)
    :param key_names: Filter names (k,)
    :return: best point, best parameters, clusters
    """
    returns = {'filter_name': [],
               'best_point': [],
               'best_parameters': [],
               'labels': []
               }
    index = 0
    image_truth_rebin = image_rebin(im_bin, rebin_factor=4, mode=rebin_mode)
    for th, name in zip(threshold_array, key_names):
        if name == 'cygno':
            std = std_map
        else:
            std = np.ones(shape=(image_batch.shape[-2], image_batch.shape[-1]))
        image_binary = image_batch[index, ...] > th*std
        image_binary_rebin = image_rebin(image_binary, rebin_factor=4, mode=rebin_mode)
        x_coord, y_coord = np.where(image_binary_rebin == True)
        X = np.array([x_coord, y_coord]).T
        index += 1
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        y = image_truth_rebin[X[:, 0], X[:, 1]].astype(float)
        eps_low = distances[distances>0].min()
        eps_high = distances.max()
        if(eps_high <= eps_low):
            eps_high = 20
            eps_low = 0.5
        space = [Integer(0, 1000, name='min_samples'),
                 Real(eps_low, eps_high , "log-uniform", name='eps')]
        @use_named_args(space)
        def objective(**params):
            m = DBSCAN(**params)
            l = m.fit(X).labels_
            result = l != -1
            sd = float(result[y == True].sum()) / sum(y == True)
            br = float((result[y == False] == False).sum()) / sum(y == False)
            f1 = 2 * (sd * br) / (sd + br)
            return 1 - f1

        res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
        m = DBSCAN(min_samples=res_gp.x[0], eps=res_gp.x[1])
        labels = m.fit(X).labels_
        returns['filter_name'].append(name)
        returns['best_point'].append(res_gp.fun)
        returns['best_parameters'].append([res_gp.x[1], res_gp.x[0]])
        returns['labels'].append(np.append(X, labels.reshape(-1,1), axis=1))
    return returns


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ResultGeneration:

    def __init__(self, folder, file_noise, run_number, bound_sup, bound_inf, roc_grid):
        self.infile = folder  # folder where there are h5 files
        self.file_noise = file_noise  # folder that has the noise files
        self.run_number = run_number  # the run number for the noise
        self.bound_sup = bound_sup    # the upper bound to remove outliers
        self.bound_inf = bound_inf    # the lower bound to remove outliers
        self.roc_grid = roc_grid      # number of points to generate roc curve
        self.input_images = None
        self.answer = {'Image_path': [], 'Image_index': [], 'Filter_name': [], 'Filter_parameter': [], 'time': [],
                       'ROC': {'full': [],
                               'rb_mean': [],
                               'rb_median': [],
                               'threshold': []},
                       'Energy': {'image_truth': [],
                                  'image_real': [],
                                  'image_after_threshold': []},
                       'Counts': {'full': [],
                                  'rb_mean': [],
                                  'rb_median': []},
                       'Clustering': {'mean': [],
                                      'median':[]}
                       }

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

    def load_images(self):
        input_data = open(self.infile).read()
        self.input_images = json.loads(json.loads(input_data))



        # get all h5 simulated files
    def get_file_names(self):
        """ get all .h5 files in folder
            input None
            output all h5 files in a list"""
        return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder

    def save_json(self, path):
        """"
        save dictionary as .json file
        """
        dumped = json.dumps(self.answer, cls=NumpyEncoder)
        with open(path + '.json', 'w') as f:
            json.dump(dumped, f)

    def get_filter_results(self, im_no_pad, image_batch, im_bin, std, im_truth, keys, roc_type, rebin=False):
        """
        Build results after image filtering
        :param im_no_pad: Image after pedestal removing
        :param image_batch: Filtered images batch
        :param im_bin: Truth image for pixels more than 0
        :param std: Std map
        :param im_truth: Truth image
        :param keys: Name of filters that were applied
        :return: roc curves, thresholds array used, pixels amount in truth images and energy curves
        """
        if 'cygno' in keys:
            cygno_index = np.where(keys == 'cygno')[0]
            image_cygno_batch = []
            #image_input, image_output, image_truth, image_std, img_truth_z, rebin, mode
            metrics_cygno_object = Metrics(im_no_pad, image_cygno_batch, im_bin, std, im_truth, rebin, roc_type)
            cy_results = metrics_cygno_object.roc_build(method='global')
            #(roc_cy, roc_rb_me_cy, roc_rb_md_cy, threshold_array_cy, count_md, count_me, cy_energy) = cy_results
            image_batch = np.delete(image_batch, cygno_index, axis=0)
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth, rebin, roc_type)
            other_results = metrics.roc_build(method='global')
            for i in range(other_results.__len__()):
                try:
                    other_results[i] = np.append(other_results[i], cy_results[i], axis=1)
                except np.AxisError:
                    if len(other_results[i]) is 0:
                        other_results[i] = None
                except ValueError:
                    try:
                        other_results[i] = np.append(other_results[i], cy_results[i].reshape(-1, 1, 1, 1), axis=1)
                        other_results[i].reshape(other_results[i].shape[0], other_results[i].shape[1])
                    except AttributeError:
                        other_results[i] = None

            #roc, roc_rb_me, roc_rb_md, threshold_array, count_md, count_me, energy = other_results
            roc = (other_results[0], other_results[1])
            roc_rb_me = (other_results[2], other_results[3])
            roc_rb_md = (other_results[4], other_results[5])
            threshold_array = other_results[6]
            count_md = other_results[7]
            count_me = other_results[8]
            energy = other_results[9]
        else:
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth, rebin, roc_type)
            results = metrics.roc_build(method='global')
            roc = (results[0], results[1])
            roc_rb_me = (results[2], results[3])
            roc_rb_md = (results[4], results[5])
            threshold_array = results[6]
            count_md = results[7]
            count_me = results[8]
            energy = results[9]
            #roc, roc_rb_me, roc_rb_md, threshold_array, count_md, count_me, energy = results
        return roc, roc_rb_me, roc_rb_md, threshold_array, count_md, count_me, energy

    def cluster_analyzer(self, roc, th, image_batch, im_bin, std, keys_array, th_mode, bg_cut):
        if th_mode == 'interpolation':
            p_choose = roc_score(roc, th, param=bg_cut)
            th_choose = [float(value) for value in p_choose['bg_constant'][0][1]]
        else:
            th_choose = abs(bg_cut - roc[1]).argmin(axis=0)
            th_choose = th[th_choose, range(th.shape[1]), 0, 0]
        for mode in ['mean', 'median']:
            self.answer['Clustering'][mode].append(apply_dbscan(image_batch,
                                                                im_bin,
                                                                std,
                                                                th_choose,
                                                                keys_array,
                                                                rebin_mode=mode))

    def image_generator(self, value_pixels, ped_map, std_map):
        """

        :param value_pixels: pixel array coords x, y, z
        :param ped_map: pedestal map used in noise generation
        :param std_map: std map used in noise generation
        :return:
        """
        image_mask = np.zeros_like(ped_map)
        image_mask[value_pixels[:, 0], value_pixels[:, 1]] = value_pixels[:, 2]
        noise_sample = np.random.normal(loc=ped_map, scale=std_map)
        im_noised = image_mask + noise_sample
        im_no_ped = im_noised - ped_map
        image_bin = image_mask > 0
        return im_no_ped, image_mask, image_bin

    def filters_apply(self, im_no_pad, filters):
        denoising_filter = DenoisingFilters(im_no_pad)
        image_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])
        for key in filters:
            if key is not 'cygno':
                filter_name = key + '_filter'
                func = getattr(DenoisingFilters, filter_name)
                params = filters[key]
                for param in params:
                    with Timer() as t1:
                        image_filtered = func(denoising_filter, *param)
                    self.answer['time'].append(str(t1.cycles))
                    image_filtered_standardized = (image_filtered - image_filtered.mean()) / image_filtered.std()
                    image_batch = np.append(image_batch, image_filtered_standardized.reshape((1,) +
                                                                                             image_filtered.shape),
                                            axis=0)
                    self.answer['Filter_name'].append(key)
                    self.answer['Filter_parameter'].append(param)
            else:
                image_batch = np.append(image_batch, im_no_pad.reshape((1,) + im_no_pad.shape), axis=0)
                self.answer['Filter_name'].append(key)
                self.answer['Filter_parameter'].append('none')
                self.answer['time'].append(str(0))
        return image_batch

    def calc_metrics(self, filters, path, filtering, rebin, clustering):
        """Apply filter on images and calculate metrics
           input filters that will be applied and folder where the output file will be saved
           output an output file with results"""
        im_ped, std = self.get_pedestal()
        self.load_images()
        file_names = self.input_images.keys()
        for file in tqdm(file_names, desc='Path applying'):
            image_dict = self.input_images[file]
            image_index = image_dict.keys()
            image_index = np.random.choice(list(image_index), 8, replace=False)
            for image_name in tqdm(image_index, desc='Image applying'):
                ipx_truth = image_dict[image_name]
                ipx_truth = np.array(ipx_truth)
                try:
                    im_no_pad, im_truth, im_bin = self.image_generator(ipx_truth, im_ped, std)
                except Exception as e:
                    print('Fail loading :' + file + '/' + image_name + '\n Error:' + str(e))
                    continue
                if filtering:
                    image_batch = self.filters_apply(im_no_pad, filters)
                    keys_array = np.array(list(filters.keys()))
                    roc_type = 'precision'
                    results = self.get_filter_results(im_no_pad, image_batch, im_bin, std, im_truth, keys_array, roc_type, rebin)
                    (rf, rme, rmd, th, count_md, count_me, energy_array) = results
                    #do_roc_plot(rf, keys_array, file)
                    if clustering is True:
                        self.cluster_analyzer(rmd,
                                              th,
                                              image_batch,
                                              im_bin,
                                              std,
                                              keys_array,
                                              th_mode='apx',
                                              bg_cut=0.9)
                    self.answer['Energy']['image_truth'].append(im_truth.sum())
                    self.answer['Energy']['image_real'].append(im_no_pad[im_bin].sum())
                    self.answer['Energy']['image_after_threshold'].append(energy_array)
                    self.answer['ROC']['full'].append(rf)
                    self.answer['ROC']['rb_mean'].append(rme)
                    self.answer['ROC']['rb_median'].append(rmd)
                    self.answer['ROC']['threshold'].append(th)
                    self.answer['Counts']['full'].append(str(im_bin.sum()))
                    self.answer['Counts']['rb_mean'].append(str(count_me))
                    self.answer['Counts']['rb_median'].append(str(count_md))
                    self.answer['Image_index'].append(image_name)
                    self.answer['Image_path'].append(file)
                    self.save_json(path)



def database_generator(path, mode='classification'):
    filter_settings = FilterSettings()
    data_folder = filter_settings.data_folder
    noise_file = filter_settings.noise_file
    run_number = filter_settings.run_number
    sup = filter_settings.sup
    inf = filter_settings.inf
    roc_grid = filter_settings.roc_grid
    data = ResultGeneration(data_folder, noise_file, run_number, sup, inf, roc_grid)
    im_ped, std = data.get_pedestal()
    data.load_images()
    file_names = data.input_images.keys()
    row_shape = []
    col_shape = []
    count_nuclear = 0
    count_eletron = 0
    for file in tqdm(file_names, desc='Path load'):
        image_dict = data.input_images[file]
        image_index = image_dict.keys()
        #image_index = np.random.choice(list(image_index), 8, replace=False)
        for image_name in tqdm(image_index, desc='Image loading'):
            ipx_truth = image_dict[image_name]
            ipx_truth = np.array(ipx_truth)
            try:
                im_no_pad, im_truth, im_bin = data.image_generator(ipx_truth, im_ped, std)
                bound_sup = im_truth.max() + 4*std.mean()
                bound_inf = -4*std.mean()
                im_no_pad = np.clip(im_no_pad, bound_inf, bound_sup)
                image_truth_cluster = regionprops(label_image=1 * im_bin, intensity_image=im_no_pad)
                cluster = image_truth_cluster[0].intensity_image
                row_len = cluster.shape[0]
                col_len = cluster.shape[1]
                row_shape.append(row_len)
                col_shape.append(col_len)
                if 'He' in file:
                    particle = 'nuclear'
                    count_nuclear += 1
                    number = count_nuclear
                else:
                    count_eletron += 1
                    particle = 'eletron'
                    number = count_eletron

                plt.imsave(path + 'train'+ '/' + particle + '/' + str(number) + '.jpg', cluster, cmap='gray')
            except Exception as e:
                print('Fail loading :' + file + '/' + image_name + '\n Error:' + str(e))
                continue
    print(np.mean(row_shape), np.mean(col_shape))



def main(filtering, rebin, clustering):
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
    data.calc_metrics(filters=filters, path=path, filtering=filtering, rebin=rebin, clustering=clustering)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--rebin", type=bool, help="If rebin will be analyzed")
    #parser.add_argument("--cluster", type=bool, help="If clustering will be analyzed")
    #args = parser.parse_args()
    #print(getattr(args, 'cluster'))
    #main(True, False, False)
    database_generator('../data/')

