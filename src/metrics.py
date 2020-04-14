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
        energy = []
        energy_real = []
        energy_should_ve = []
        thresholds = []
        xs, ys = np.where(self._image_truth == 1)
        xb, yb = np.where(self._image_truth == 0)
        energy_truth = self.energy_calc(xs, ys)
        energy_real_truth = self.energy_calc(xs, ys, kind='real')
        for i in range(0, grid + 1):
            thr = bound_inf + i * step
            try:
                thr = thr.reshape(-1, 1, 1)
            except AttributeError:
                thr = thr
            result = self._image_output > (thr * px_thr)
            thresholds.append(thr)
            intersection = result[:, xs, ys]
            energy_should_ve_temp = []
            energy_temp = []
            energy_real_temp = []
            for image_n in range(intersection.shape[0]):
                xi, yi = xs[intersection[image_n, :]], ys[intersection[image_n, :]]
                xsh, ysh = np.where(result[image_n, :, :] == True)
                energy_should_ve_temp.append(self.energy_calc(xsh, ysh, 'real'))
                energy_temp.append(self.energy_calc(xi, yi)-energy_truth)
                energy_real_temp.append(self.energy_calc(xi, yi, kind='real') - energy_real_truth)
            energy.append(energy_temp)
            energy_real.append(energy_real_temp)
            energy_should_ve.append(energy_should_ve_temp)
            signal_pixels_eff = result[:, xs, ys].sum(axis=1) / len(xs)
            background_pixels_eff = (result[:, xb, yb] == False).sum(axis=1) / len(xb)
            sg_ef.append(signal_pixels_eff)
            bg_ef.append(background_pixels_eff)
        return (np.array(sg_ef), np.array(bg_ef)), np.array(energy), np.array(energy_real), np.array(energy_should_ve), np.array(thresholds)

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













