import numpy as np
import h5py
import glob
from settings import FilterSettings
from filters import DenoisingFilters
from metrics import Metrics
import json
from hwcounter import Timer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import pandas as pd

SEED = pd.read_csv('../data/lut_seed.csv')
SEED.index = SEED['Unnamed: 0']

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

def save_json(obj, path):
    """"
    save dictionary as .json file
    """
    dumped = json.dumps(obj, cls=NumpyEncoder)
    with open(path + '.json', 'w') as f:
        json.dump(dumped, f)


class ResultGeneration:

    def __init__(self, folder, file_noise, run_number, bound_sup, bound_inf, roc_grid):
        self.infile = folder  # folder where there are h5 files
        self.file_noise = file_noise  # folder that has the noise files
        self.run_number = run_number  # the run number for the noise
        self.bound_sup = bound_sup    # the upper bound to remove outliers
        self.bound_inf = bound_inf    # the lower bound to remove outliers
        self.roc_grid = roc_grid      # number of points to generate roc curve
        self.input_images = None
        self.answer = {'Image_path': [],
                       'Image_index': [],
                       'Filter_name': [],
                       'Filter_parameter': [],
                       'time': [],
                       'ROC': {'full': [],
                               'threshold': []},
                       'Energy': {'image_truth': [],
                                  'image_real': [],
                                  'image_after_threshold': []},
                       'Counts': []
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



    #     # get all h5 simulated files
    # def get_file_names(self):
    #     """ get all .h5 files in folder
    #         input None
    #         output all h5 files in a list"""
    #     return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder



    # def get_filter_results(self, im_no_pad, image_batch, im_bin, std, im_truth, keys, thresholds, mode='eval'):
    #     """
    #     Build results after image filtering
    #     :param im_no_pad: Image after pedestal removing
    #     :param image_batch: Filtered images batch
    #     :param im_bin: Truth image for pixels more than 0
    #     :param std: Std map
    #     :param im_truth: Truth image
    #     :param keys: Name of filters that were applied
    #     :param grid: Number of samples roc will be built
    #     :return: roc curves, thresholds array used, pixels amount in truth images and energy curves
    #     """
    #     if ('cygno' in keys) & (len(keys) > 1):
    #         cygno_index = np.where(keys == 'cygno')[0]
    #         image_cygno_batch = []
    #         metrics_cygno_object = Metrics(im_no_pad, image_cygno_batch, im_bin, std, im_truth, thresholds)
    #         cy_results = metrics_cygno_object.roc_build(self.roc_grid, method='local')
    #         image_batch = np.delete(image_batch, cygno_index, axis=0)
    #         metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
    #         other_results = metrics.roc_build(self.roc_grid, method='global')
    #         for i in range(other_results.__len__()):
    #             try:
    #                 other_results[i] = np.append(other_results[i], cy_results[i], axis=1)
    #             except np.AxisError:
    #                 if len(other_results[i]) is 0:
    #                     other_results[i] = None
    #             except ValueError:
    #                 try:
    #                     other_results[i] = np.append(other_results[i], cy_results[i].reshape(-1, 1, 1, 1), axis=1)
    #                     other_results[i].reshape(other_results[i].shape[0], other_results[i].shape[1])
    #                 except AttributeError:
    #                     other_results[i] = None
    #
    #         roc = (other_results[0], other_results[1])
    #         threshold_array = other_results[2]
    #         energy = other_results[3]
    #     else:
    #         metrics = Metrics(im_no_pad, image_batch, im_bin, std, im_truth)
    #         if ('cygno' in keys):
    #             results = metrics.roc_build(self.roc_grid, method='local')
    #         else:
    #             results = metrics.roc_build(self.roc_grid, method='global')
    #         roc = (results[0], results[1])
    #         threshold_array = results[2]
    #         energy = results[3]
    #     return roc, threshold_array, energy



    def image_generator(self, value_pixels, ped_map, std_map, seed=None):
        """

        :param value_pixels: pixel array coords x, y, z
        :param ped_map: pedestal map used in noise generation
        :param std_map: std map used in noise generation
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        image_mask = np.zeros_like(ped_map)
        image_mask[value_pixels[:, 0], value_pixels[:, 1]] = value_pixels[:, 2]
        noise_sample = np.random.normal(loc=ped_map, scale=std_map)
        im_noised = image_mask + noise_sample
        im_no_ped = im_noised - ped_map
        image_bin = image_mask > 0
        return im_no_ped, image_mask, image_bin

    def filters_apply(self, im_no_pad, im_bin, filters, im_std):
        xs, ys = np.where(im_bin == True)
        denoising_filter = DenoisingFilters(im_no_pad)
        image_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])
        std_batch = np.empty([0, im_no_pad.shape[0], im_no_pad.shape[1]])

        # TODO: Change bound inf and sup to max and min xs,ys
        bound_inf = []
        bound_sup = []
        for key in filters:
            if key is not 'cygno':
                filter_name = key + '_filter'
                try:
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
                        std_batch = np.append(std_batch, np.ones(((1,) + image_filtered.shape)), axis=0)
                        bound_inf.append(image_filtered_standardized[xs, ys].min())
                        bound_sup.append(image_filtered_standardized[xs, ys].max())
                        self.answer['Filter_name'].append(key)
                        self.answer['Filter_parameter'].append(param)
                    pass
                except AttributeError:
                    print('Filter {} has not been found'.format(key))
                    continue
            else:
                im_filtered = im_no_pad
                # lower = im_filtered.min()
                # upper = 2*im_filtered[xs, ys].max()
                # #im_filtered[im_filtered>(1.5*upper)] = 0
                # im_filtered = np.clip(im_filtered, lower, upper)
                image_batch = np.append(image_batch, im_filtered.reshape((1,) + im_filtered.shape), axis=0)
                std_batch = np.append(std_batch, im_std.reshape((1,) + im_std.shape), axis=0)
                bound_inf.append(-4)
                bound_sup.append(60)
                self.answer['Filter_name'].append(key)
                self.answer['Filter_parameter'].append('none')
                self.answer['time'].append(str(0))
        return image_batch, std_batch, bound_inf, bound_sup

    def build_results(self, im_no_pad, im_bin, image_batch, std_batch, im_truth, threshold, mode='build'):
        metrics = Metrics(im_no_pad, image_batch, im_bin, std_batch, im_truth, threshold)
        if mode == 'build':
            results = metrics.roc_build(self.roc_grid)
            roc = (results[0], results[1])
            threshold_array = results[2]
            energy = results[3]
            return roc, threshold_array, energy
        else:
            _, image = metrics.roc_build(self.roc_grid, return_last=True)
            return image



    def calc_metrics(self, filters, path, images):
        """Apply filter on images and calculate metrics
           input filters that will be applied and folder where the output file will be saved
           output an output file with results"""
        im_ped, std = self.get_pedestal()
        self.load_images()
        file_names = self.input_images.keys()
        for file in tqdm(file_names, desc='Path applying'):
            image_dict = self.input_images[file]
            image_index = image_dict.keys()
            image_index = np.random.permutation(list(image_index))[0:images]
            for image_name in tqdm(image_index, desc='Image applying'):
                ipx_truth = image_dict[image_name]
                ipx_truth = np.array(ipx_truth)
                try:
                    im_no_pad, im_truth, im_bin = self.image_generator(ipx_truth, im_ped, std)
                except Exception as e:
                    print('Fail loading :' + file + '/' + image_name + '\n Error:' + str(e))
                    continue
                image_batch, std_batch, inf, sup = self.filters_apply(im_no_pad, im_bin, filters, std)
                threshold = [np.array(inf), np.array(sup)]
                results = self.build_results(im_no_pad, im_bin, image_batch, std_batch, im_truth, threshold)
                # keys_array = np.array(list(filters.keys()))
                # results = self.get_filter_results(im_no_pad,
                #                                   image_batch,
                #                                   im_bin,
                #                                   std,
                #                                   im_truth,
                #                                   keys_array,
                #                                   thresholds=None,
                #                                   mode=mode)
                (rf, th, energy_array) = results
                #do_roc_plot(rf, keys_array, file)
                self.answer['Energy']['image_truth'].append(im_truth.sum())
                self.answer['Energy']['image_real'].append(im_no_pad[im_bin].sum())
                self.answer['Energy']['image_after_threshold'].append(energy_array)
                self.answer['ROC']['full'].append(rf)
                self.answer['ROC']['threshold'].append(th)
                self.answer['Counts'].append(str(im_bin.sum()))
                self.answer['Image_index'].append(image_name)
                self.answer['Image_path'].append(file)
                save_json(self.answer, path)


    def image_from_metrics(self, filter_files, images):
        output_dict = {'path': [], 'image': [], 'filter': [], 'points': []}
        dataframe, packs = filter_files
        input_data = open(dataframe).read()
        params = json.loads(input_data)#json.loads(json.loads(input_data)[0])
        out_dict = {}
        threshold_value = []
        for dict_p in params:
            out_dict.update({dict_p['filter']: [[dict_p['parameter']]]})
            threshold_value.append(dict_p['threshold'])

        im_ped, std = self.get_pedestal()
        self.load_images()
        #file_names = list(self.input_images.keys())[0:3]
        file_names = packs
        for file in tqdm(file_names, desc='Path applying'):
            image_dict = self.input_images[file]
            image_index = image_dict.keys()
            image_index = np.random.permutation(list(image_index))[0:images]
            image_counter = 0
            for image_name in tqdm(image_index, desc='Image applying'):
                ipx_truth = image_dict[image_name]
                ipx_truth = np.array(ipx_truth)
                try:
                    seed = SEED[file][image_name]
                    im_no_pad, im_truth, im_bin = self.image_generator(ipx_truth, im_ped, std, seed=seed)
                except Exception as e:
                    print('Fail loading :' + file + '/' + image_name + '\n Error:' + str(e))
                    continue
                image_batch, std_batch, inf, sup = self.filters_apply(im_no_pad, im_bin, out_dict, std)
                threshold = [np.array(threshold_value), np.array(threshold_value)]
                #threshold = [np.array(inf), np.array(sup)]
                results = self.build_results(im_no_pad, im_bin, image_batch, std_batch, im_truth, threshold, mode='eval')
                ## add truth to results
                results = np.append(results, im_bin.reshape((1,) + im_bin.shape), axis=0)
                index_matrix = np.array(np.where(results == True)).T
                split_results = np.split(index_matrix, np.cumsum(np.unique(index_matrix[:, 0], return_counts=True)[1])[:-1])
                applied_filters = list(out_dict.keys())
                applied_filters.append('truth')
                for i, pts in list(enumerate(split_results)):
                    output_dict['path'].append(file)
                    output_dict['image'].append(image_name)
                    output_dict['filter'].append(applied_filters[i])
                    output_dict['points'].append(pts)

                save_json(output_dict, '../data/run_clustering')
                image_counter+=1
            print('Path {} has been done with {} images processed'.format(file, image_counter))
        print('\n Process has been finished')
def main(mode):
    filter_settings = FilterSettings()
    data_folder = filter_settings.data_folder
    noise_file = filter_settings.noise_file
    run_number = filter_settings.run_number
    sup = filter_settings.sup
    inf = filter_settings.inf
    roc_grid = filter_settings.roc_grid
    n_samples = filter_settings.nsamples
    data = ResultGeneration(data_folder, noise_file, run_number, sup, inf, roc_grid)
    if mode == 'build':
        filters = filter_settings.filters
        path = filter_settings.output_file_path + filter_settings.output_file_name
        data.calc_metrics(filters=filters, path=path, images=n_samples)
    elif mode == 'eval':
        pre_processing_params = filter_settings.preprocessing_params
        data.image_from_metrics(pre_processing_params, n_samples)
    #thresholds = np.array(filter_settings.thresholds)
    #





if __name__ == "__main__":
    main('eval')


