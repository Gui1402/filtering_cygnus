import numpy as np
from settings import FilterSettings
from skimage.morphology import skeletonize
from skimage.morphology import label
from skimage.measure import regionprops, regionprops_table
from scipy.interpolate import interp1d
from skimage.filters import laplace
from skimage.measure import find_contours
from skimage.transform import rotate
from skimage.morphology import disk


# def image_rebin(a, shape, mode):
#     sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
#     if mode == 'mean':
#         return a.reshape(sh).mean(-1).mean(1)
#     elif mode == 'median':
#         return np.median(np.median(a.reshape(sh), axis=-1), axis=1)
#
# def arrrebin(img, rebin, mode='mean'):
#     newshape = int(2048 / rebin)
#     img_rebin = image_rebin(img, (newshape, newshape), mode)
#     return img_rebin

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



def cluster_length(image_cluster):
    skeleton_lee = skeletonize(image_cluster, method='lee') / 255
    return skeleton_lee.sum()


def cluster_features(img_clusters):
    clusters_info = regionprops(img_clusters)
    cluster_table = {"xc": [],
                     "yc": [],
                     "length": [],
                     "width": [],
                     "area": [],
                     "angle": [],
                     "perimeter": []}
    output_cluster = []
    for cluster in clusters_info:
        area = cluster.area
        if area > 12:
            centroid = cluster.centroid
            cluster_table["xc"].append(centroid[0])
            cluster_table["yc"].append(centroid[1])
            length = cluster_length(cluster.image)
            cluster_table["length"].append(length)
            cluster_table["area"].append(area)
            cluster_table["width"].append(area / length)
            cluster_table["angle"].append(cluster.orientation)
            cluster_table["perimeter"].append(cluster.perimeter)
            output_cluster.append(cluster)

    return cluster_table, output_cluster


class Metrics:

    def __init__(self, image_input, image_output, image_truth, image_std, img_truth_z):
        self._image_input = image_input
        self._image_output = image_output
        self._image_truth = image_truth
        self._image_std = image_std
        self._image_truth_z = img_truth_z

    def find_a_in_b(self, a, b):
        nrows, ncols = a.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [a.dtype]}

        c = np.intersect1d(a.view(dtype), b.view(dtype))

        # This last bit is optional if you're okay with "C" being a structured array...
        c = c.view(a.dtype).reshape(-1, ncols)
        return c

    def roc_build(self, method='global'):
        fs = FilterSettings()
        if len(self._image_output) == 0:
            method = 'local'
            bound_sup = 5
            bound_inf = -7
            self._image_output = self._image_input.reshape((1,) + self._image_input.shape)
        else:
            bound_sup = 1.3 * np.percentile(np.percentile(self._image_output, 99, axis=2), 99, 1)
            bound_inf = 0.7 * np.percentile(np.percentile(self._image_output, 1, axis=2), 1, 1)

        if method == 'local':
            px_thr = np.broadcast_to(self._image_std, self._image_output.shape)
        else:
            px_thr = np.ones_like(self._image_output)
        grid = fs.roc_grid
        step = (bound_sup - bound_inf) / grid
        sg_ef = []
        bg_ef = []
        sg_ef_mean = []
        bg_ef_mean = []
        sg_ef_median = []
        bg_ef_median = []
        energy_intersect = []
        thresholds = []
        xs, ys = np.where(self._image_truth == 1)
        xb, yb = np.where(self._image_truth == 0)
        im_truth_rebin_mean = image_rebin(self._image_truth, rebin_factor=4, mode='mean')
        im_truth_rebin_median = image_rebin(self._image_truth, rebin_factor=4, mode='median')
        xs_me, ys_me = np.where(im_truth_rebin_mean == 1)
        xs_md, ys_md = np.where(im_truth_rebin_median == 1)
        xb_me, yb_me = np.where(im_truth_rebin_mean == 0)
        xb_md, yb_md = np.where(im_truth_rebin_median == 0)
        for i in range(0, grid + 1):
            thr = bound_inf + i * step
            try:
                thr = thr.reshape(-1, 1, 1)
            except AttributeError:
                thr = thr
            images_after_threshold = self._image_output > (thr * px_thr)
            mean_rebin = image_rebin(images_after_threshold, rebin_factor=4, mode='mean')
            median_rebin = image_rebin(images_after_threshold, rebin_factor=4, mode='median')

            full_sg_eff, full_bg_eff, energy = self.roc_outputs(images_after_threshold, xs, ys, xb, yb)
            me_sg_eff, me_bg_eff, _ = self.roc_outputs(mean_rebin, xs_me, ys_me, xb_me, yb_me)
            md_sg_eff, md_bg_eff, _ = self.roc_outputs(median_rebin, xs_md, ys_md, xb_md, yb_md)
            sg_ef.append(full_sg_eff)
            bg_ef.append(full_bg_eff)
            sg_ef_mean.append(me_sg_eff)
            bg_ef_mean.append(me_bg_eff)
            sg_ef_median.append(md_sg_eff)
            bg_ef_median.append(md_bg_eff)
            energy_intersect.append(energy)
            thresholds.append(thr)
        return (np.array(sg_ef), np.array(bg_ef)), (np.array(sg_ef_mean), np.array(bg_ef_mean)), \
               (np.array(sg_ef_median), np.array(bg_ef_median)), \
                np.array(thresholds), len(xs_md), len(xs_me), np.array(energy_intersect)

    def roc_outputs(self, result, xs, ys, xb, yb):
        index_matrix = np.array(np.where(result[:, xs, ys] == True)).T
        index_matrix = np.append(index_matrix, np.array((xs[index_matrix[:, 1]], ys[index_matrix[:, 1]])).T,
                                 axis=1)[:, [0, 2, 3]]
        z_values = self._image_input[index_matrix[:, 1], index_matrix[:, 2]]
        z_values_split = np.split(z_values, np.cumsum(np.unique(index_matrix[:, 0], return_counts=True)[1])[:-1])
        energy = list(map(sum, z_values_split))
        signal_pixels_eff = result[:, xs, ys].sum(axis=1) / len(xs)
        background_pixels_eff = (result[:, xb, yb] == False).sum(axis=1) / len(xb)
        return signal_pixels_eff, background_pixels_eff, energy

    def calc_auc(self, roc):
        x = roc[0, :]
        y = roc[1, :]
        auc = np.sum(np.abs(np.diff(x)) * y[:-1])
        return auc

    def snr_calc(self):
        output = []
        truth_spx, truth_spy, truth_spz = np.where(self._image_truth==True)  # pixels pertencentes a ROI definida
        truth_bpx, truth_bpy, truth_bpz = np.where(self._image_truth==False)  # pixels fora da ROI definida
        for index in np.unique(truth_spz):
            isx, isy = truth_spx[truth_spz == index], truth_spy[truth_spz == index]
            ibx, iby = truth_bpx[truth_bpz == index], truth_bpy[truth_bpz == index]
            en_sg = sum(self._image_output[isx, isy, index] ** 2)
            en_bg = sum(self._image_output[ibx, iby, index] ** 2)
            output.append(en_sg / en_bg)
        return output

    def energy_calc(self, xx, yy, kind='truth'):
        if kind == 'truth':
            return self._image_truth_z[xx, yy].sum()
        else:
            return self._image_input[xx, yy].sum()

    @staticmethod
    def roc_score(roc, threshold, param=0.9):
        threshold = threshold[:, :, 0, 0]
        n_filters = threshold.shape[1]
        x = roc[0]
        y = roc[1]
        result = {'f1_constant': [],
                  'sg_constant': [],
                  'bg_constant': []}

        f1_score = (2*x*y)/(x+y)
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


class ClusterMetrics:
    def __init__(self, image_bin, image_truth, image_intensities):
        self.image_bin = image_bin
        self.image_truth = image_truth
        self.image_intensities = image_intensities
        self.image_truth_cluster = None
        self.image_bin_cluster = None

    def get_labels(self):
        self.image_truth_cluster = 1*self.image_truth
        self.image_bin_cluster = label(self.image_bin)

    def get_clusters_features(self):
        self.get_labels()
        table_truth, cluster_truth = cluster_features(self.image_truth_cluster)
        table_real, cluster_real = cluster_features(self.image_bin_cluster)
        return table_truth, table_real, cluster_truth, cluster_real













