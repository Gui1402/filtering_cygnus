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

    def roc_build(self):
        fs = FilterSettings()
        bound_sup = fs.sup
        bound_inf = fs.inf
        grid = fs.roc_grid
        output = []
        sx, sy = np.where(self._image_truth == 1)
        bx, by = np.where(self._image_truth == 0)
        sg_ef = []
        bg_ef = []
        for thr in np.linspace(bound_inf, bound_sup, grid):
            sxx, syy = np.where(self._image_output >= thr*self._image_std)
            s_pixels = self.find_a_in_b(np.stack((sxx, syy), axis=1),
                                        np.stack((sx, sy), axis=1))
            im_thr = self._image_output >= thr*self._image_std
            sg_ef.append(sum(im_thr[sx, sy] == 1) / len(sx))
            bg_ef.append(sum(im_thr[bx, by] == 0) / len(bx))
        output.append([sg_ef, bg_ef])
        return np.array(output)

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
