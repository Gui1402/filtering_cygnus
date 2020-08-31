import numpy as np
from scipy.interpolate import interp1d

def roc_score(roc, threshold, param=0.9):
    threshold = threshold[:, :, 0, 0]
    n_filters = threshold.shape[1]
    x = roc[0]
    y = roc[1]
    result = {'f1_constant': [],
              'sg_constant': [],
              'bg_constant': []}

    f1_score = (2 * x * y) / (x + y)
    best_f1 = f1_score.max(axis=0)
    best_f1_indexes = f1_score.argmax(axis=0)
    best_thresholds = threshold[list(best_f1_indexes), range(0, n_filters)]
    result['f1_constant'].append([best_f1, best_thresholds])
    sg_constant = []
    th_sg_constant = []
    th_bg_constant = []
    bg_constant = []
    for filter in range(n_filters):
        sg_eff, bg_eff, threshold_value = x[:, filter], y[:, filter], threshold[:, filter]
        csb = interp1d(sg_eff, bg_eff, fill_value=True)
        cbs = interp1d(bg_eff, sg_eff, fill_value=True)
        cst = interp1d(sg_eff, threshold_value, fill_value=True)
        cbt = interp1d(bg_eff, threshold_value, fill_value=True)
        sg_constant.append(csb(param))
        bg_constant.append(cbs(param))
        th_sg_constant.append(cst(param))
        th_bg_constant.append(cbt(param))
    result['sg_constant'].append([sg_constant, th_sg_constant])
    result['bg_constant'].append([bg_constant, th_bg_constant])
    return result

def image_rebin(image_input, rebin_factor, mode='mean', threshold=0):
    image_shape = image_input.shape
    if len(image_shape) > 2:
        new_shape = (-1, image_shape[1]//rebin_factor, rebin_factor, image_shape[2]//rebin_factor, rebin_factor)
        ax_return = 2
    else:
        new_shape = (image_shape[0]//rebin_factor, rebin_factor, image_shape[1]//rebin_factor, rebin_factor)
        ax_return = 1
    if mode == 'mean':
        return image_input.reshape(new_shape).mean(-1).mean(axis=ax_return) > threshold
    elif mode == 'median':
        return np.median(np.median(image_input.reshape(new_shape), axis=-1), axis=ax_return) > threshold







class Metrics:

    def __init__(self, image_input, image_output, image_truth, image_std, img_truth_z):
        self._image_input = image_input
        self._image_output = image_output
        self._image_truth = image_truth
        self._image_std = image_std
        self._image_truth_z = img_truth_z

    def roc_build(self, grid, method='global'):
        xs, ys = np.where(self._image_truth == 1)
        # TODO: put std ped as an argument, each filter will pass our own std ped (all ones for all except cygno)
        if len(self._image_output) == 0:
            method = 'local'
            bound_sup = 4
            bound_inf = -20
            self._image_output = self._image_input.reshape((1,) + self._image_input.shape)
        else:
            # TODO: here I shouldn't have to do this
            bound_inf = self._image_output[:, xs, ys].min(axis=1)
            bound_sup = self._image_output[:, xs, ys].max(axis=1)
        if method == 'local':
            px_thr = np.broadcast_to(self._image_std, self._image_output.shape)
        else:
            px_thr = np.ones_like(self._image_output)

        step = (bound_sup - bound_inf) / grid
        sg_ef = []
        bg_ef = []
        energy_intersect = []
        thresholds = []
        for i in range(0, grid + 1):
            thr = bound_inf + i * step
            try:
                thr = thr.reshape(-1, 1, 1)
            except AttributeError:
                thr = thr
            images_after_threshold = self._image_output > (thr * px_thr)
            full_sg_eff, full_bg_eff, energy = self.roc_outputs(images_after_threshold,
                                                                xs, ys)
            sg_ef.append(full_sg_eff)
            bg_ef.append(full_bg_eff)
            energy_intersect.append(energy)
            thresholds.append(thr)
        return [np.nan_to_num(np.array(sg_ef)),
                np.nan_to_num(np.array(bg_ef)),
                np.array(thresholds),
                np.array(energy_intersect)]

    def roc_outputs(self, result, xs, ys):
        energy = []
        intersect_maps = result & self._image_truth
        for intersect_id in range(intersect_maps.shape[0]):
            energy.append(np.sum(self._image_input[intersect_maps[intersect_id, :, :]]))
        # index_matrix = np.array(np.where(result[:, xs, ys] == True)).T
        # index_matrix = np.append(index_matrix, np.array((xs[index_matrix[:, 1]], ys[index_matrix[:, 1]])).T,
        #                          axis=1)[:, [0, 2, 3]]
        # ## TODO: Review this energy calc
        # z_values = self._image_input[index_matrix[:, 1], index_matrix[:, 2]]
        # z_values_split = np.split(z_values, np.cumsum(np.unique(index_matrix[:, 0], return_counts=True)[1])[:-1])
        # index_energy = np.unique(index_matrix[:, 0])
        # energy = list(map(sum, z_values_split))
        # energy_dict = dict(zip(index_energy, energy))
        # energy = [energy_dict.get(i, 0) for i in range(result.shape[0])]
        background_pixels_eff = result[:, xs, ys].sum(axis=1)/result.sum(axis=1).sum(axis=1)
        signal_pixels_eff = result[:, xs, ys].sum(axis=1) / len(xs)

        return signal_pixels_eff, background_pixels_eff, energy

    def image_evaluator(self, threshold_value, method='global'):
        xs, ys = np.where(self._image_truth == 1)
        if method == 'local':
            px_thr = np.broadcast_to(self._image_std, self._image_output.shape)
        else:
            px_thr = np.ones_like(self._image_output)
        try:
            threshold_value = threshold_value.reshape(-1, 1, 1)
        except AttributeError:
            threshold_value = threshold_value

        images_after_threshold = self._image_output > (threshold_value * px_thr)
        recall, precision, energy = self.roc_outputs(images_after_threshold, xs, ys)

        return images_after_threshold, recall, precision, energy
















