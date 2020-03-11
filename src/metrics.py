import numpy as np
from settings import FilterSettings


class Metrics:

    def __init__(self, image_input, image_output, image_truth, image_std):
        self._image_input = image_input
        self._image_output = image_output
        self._image_truth = image_truth
        self._image_std = image_std

    def find_a_in_b(self, a, b):
        nrows, ncols = a.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [a.dtype]}

        c = np.intersect1d(a.view(dtype), b.view(dtype))

        # This last bit is optional if you're okay with "C" being a structured array...
        c = c.view(a.dtype).reshape(-1, ncols)
        return c

    def roc_build(self, method='local'):
        fs = FilterSettings()
        bound_sup = np.percentile(np.percentile(self._image_output, 95, axis=2), 95, 1)
        bound_inf = np.percentile(np.percentile(self._image_output, 5, axis=2), 5, 1)
        if method == 'local':
            px_thr = np.broadcast_to(self._image_std, self._image_output.shape)
        else:
            px_thr = np.ones_like(self._image_output)
        grid = fs.roc_grid
        step = (bound_sup - bound_inf) / grid
        sg_ef = []
        bg_ef = []
        energy = []
        xs, ys = np.where(self._image_truth == 1)
        xb, yb = np.where(self._image_truth == 0)
        energy_truth = self.energy_calc(xs, ys)
        for i in range(0, grid + 1):
            thr = bound_inf + i * step
            thr = thr.reshape(-1, 1, 1)
            result = self._image_output > (thr * px_thr)
            intersection = result[:, xs, ys]
            energy_temp = []
            for image_n in range(intersection.shape[0]):
                xi, yi = xs[intersection[image_n, :]], ys[intersection[image_n, :]]
                energy_temp.append(energy_truth-self.energy_calc(xi, yi))
            energy.append(energy_temp)
            signal_pixels_eff = result[:, xs, ys].sum(axis=1) / len(xs)
            background_pixels_eff = (result[:, xb, yb] == False).sum(axis=1) / len(xb)
            sg_ef.append(signal_pixels_eff)
            bg_ef.append(background_pixels_eff)
        return (np.array(sg_ef), np.array(bg_ef)), np.array(energy)

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

    def energy_calc(self, xx, yy):
        return self._image_input[xx, yy].sum()