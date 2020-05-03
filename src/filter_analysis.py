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
        space = [Integer(0, 1000, name='min_samples'),
                 Real(distances[distances>0].min(), distances.max(), "log-uniform", name='eps')]
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
        self.folder = folder  # folder where there are h5 files
        self.file_noise = file_noise  # folder that has the noise files
        self.run_number = run_number  # the run number for the noise
        self.bound_sup = bound_sup    # the upper bound to remove outliers
        self.bound_inf = bound_inf    # the lower bound to remove outliers
        self.roc_grid = roc_grid      # number of points to generate roc curve
        self.answer = {'Image_index': [], 'Filter_name': [], 'Filter_parameter': [], 'time': [],
                       'ROC': {'full': [],
                               'rb_mean': [],
                               'rb_median': [],
                               'threshold': []},
                       'Energy': {'image_truth': [],
                                  'image_no_ped': [],
                                  'image_after_threshold': []},
                       'Counts': {'full': [],
                                  'rb_mean': [],
                                  'rb_median': []},
                       'Clustering': []
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

    # get all h5 simulated files
    def get_file_names(self):
        """ get all .h5 files in folder
            input None
            output all h5 files in a list"""
        return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder

    def get_filter_results(self, im_no_pad, image_batch, im_bin, std, im_truth, keys):
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
            metrics_cygno_object = Metrics(im_no_pad, image_cygno_batch, im_bin, std, im_truth)
            cy_results = metrics_cygno_object.roc_build(method='global')
            (roc_cy, roc_rb_me_cy, roc_rb_md_cy, threshold_array_cy, count_md, count_me, cy_energy) = cy_results
            image_batch = np.delete(image_batch, cygno_index, axis=0)
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
            other_results = metrics.roc_build(method='global')
            (roc, roc_rb_me, roc_rb_md, threshold_array, _, _, energy) = other_results
            roc0 = np.append(roc_cy[0], roc[0], axis=1)
            roc1 = np.append(roc_cy[1], roc[1], axis=1)
            roc = (roc0, roc1)

            roc_rb_me_0 = np.append(roc_rb_me_cy[0], roc_rb_me[0], axis=1)
            roc_rb_me_1 = np.append(roc_rb_me_cy[1], roc_rb_me[1], axis=1)
            roc_rb_me = (roc_rb_me_0, roc_rb_me_1)

            roc_rb_md_0 = np.append(roc_rb_md_cy[0], roc_rb_md[0], axis=1)
            roc_rb_md_1 = np.append(roc_rb_md_cy[1], roc_rb_md[1], axis=1)
            roc_rb_md = (roc_rb_md_0, roc_rb_md_1)
            threshold_array = np.append(threshold_array_cy.reshape(-1, 1, 1, 1), threshold_array, axis=1)
            energy = np.append(energy, cy_energy, axis=1)

        else:
            metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
            results = metrics.roc_build(method='global')
            roc, roc_rb_me, roc_rb_md, threshold_array, count_md, count_me, energy = results
        return roc, roc_rb_me, roc_rb_md, threshold_array, count_md, count_me, energy



    def calc_metrics(self, filters, path):
        """Apply filter on images and calculate metrics
           input filters that will be applied and folder where the output file will be saved
           output an output file with results"""
        full_files = self.get_file_names()
        im_ped, std = self.get_pedestal()
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
                im_no_pad = im_real - im_ped
                im_bin = im_truth > 0
                #self.answer['count'].append(str(im_bin.sum()))
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
                        image_batch = np.append(image_batch, im_no_pad.reshape((1, ) + im_no_pad.shape), axis=0)
                        self.answer['Filter_name'].append(key)
                        self.answer['Filter_parameter'].append('none')
                        self.answer['time'].append(str(0))

                keys_array = np.array(list(filters.keys()))
                results = self.get_filter_results(im_no_pad, image_batch, im_bin, std, im_truth, keys_array)
                (rf, rme, rmd, th, count_md, count_me, energy_array) = results
                #p_choose = roc_score(rmd, th, param=0.90)
                #th_choose = [float(value) for value in p_choose['bg_constant'][0][1]]
                th_choose = abs(0.9 - rmd[1]).argmin(axis=0)
                th_choose = th[th_choose, range(th.shape[1]), 0, 0]
                db_result = apply_dbscan(image_batch, im_bin, std, th_choose, keys_array, rebin_mode='median')
                self.answer['Energy']['image_truth'] = im_truth.sum()
                self.answer['Energy']['image_real'] = im_no_pad.sum()
                self.answer['Energy']['image_after_threshold'] = energy_array
                self.answer['ROC']['full'].append(rf)
                self.answer['ROC']['rb_mean'].append(rme)
                self.answer['ROC']['rb_median'].append(rmd)
                self.answer['ROC']['threshold'].append(th)
                self.answer['Counts']['full'].append(str(im_bin.sum()))
                self.answer['Counts']['rb_mean'].append(str(count_me))
                self.answer['Counts']['rb_median'].append(str(count_md))
                self.answer['Image_index'].append(image_index)
                self.answer['Clustering'].append(db_result)
                bar.next()
                b = time()-a
                remaining = (n_images-image_index)*b
                print('\nEstimated time per image = ' + str(b) + '\n'
                      'Remaining (apx) = ' + str(round(remaining/60))+' minutes \r',)
                self.save_json(path)

            bar.finish()

    def save_json(self, path):
        """"
        save dictionary as .json file
        """
        dumped = json.dumps(self.answer, cls=NumpyEncoder)
        with open(path + '.json', 'w') as f:
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
    filters = filter_settings.best_filters
    path = filter_settings.output_file_path + filter_settings.output_file_name
    data.calc_metrics(filters=filters, path=path)


if __name__ == "__main__":
    main()

