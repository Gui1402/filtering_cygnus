import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.filters import median
from skimage.morphology import disk
from skimage import restoration
from scipy import signal
import h5py
import glob
import itertools
from progress.bar import Bar
from time import time



# Criacao de uma mascara gaussiana
def gkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / np.sum(gkern2d)


# criando uma funcao de filtragem multidimensional
def linear_filtering(img, maskSize, filtType):
    mask = np.zeros((maskSize, maskSize, 1))
    if filtType == 'gauss':
        # relacao W ~ 3*sigma
        sigma = maskSize / 3
        mask[:, :, 0] = gkern(maskSize, sigma)

    elif filtType == 'mean':
        # criando uma mascara de media
        mask[:, :, 0] = (1 / (maskSize ** 2)) * np.ones((maskSize, maskSize))
    elif filtType == 'laplacian':
        mask[:, :, 0] = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    elif filtType == 'median':
        deconvolved_img = np.zeros_like(img)
        for i in range(0, img.shape[2]):
            deconvolved_img[:, :, i] = median(img[:, :, i], disk(maskSize))
            #deconvolved_img[:, :, i] = restoration.wiener(img[:, :, i], np.ones((1, 1)), maskSize/3)
    else:
        print("filter has not found")
        # se nenhum filtro for escolhido convolui com um impulso
        mask[1, 1, :] = 1
    mask[:, :, 0] = np.rot90(np.rot90(mask[:, :, 0]))
    # convolui com filtro rotacionado de 180 graus
    if(filtType == 'median'):
        return deconvolved_img
    else:
        return convolve(img, mask, mode='wrap')


def snr_calc(im_filt, im_bin):
    output = []
    truth_spx, truth_spy, truth_spz = np.where(im_bin == True)  # pixels pertencentes a ROI definida
    truth_bpx, truth_bpy, truth_bpz = np.where(im_bin == False)  # pixels fora da ROI definida
    for index in np.unique(truth_spz):
        isx, isy = truth_spx[truth_spz == index], truth_spy[truth_spz == index]
        ibx, iby = truth_bpx[truth_bpz == index], truth_bpy[truth_bpz == index]
        en_sg = sum(im_filt[isx, isy, index] ** 2)
        en_bg = sum(im_filt[ibx, iby, index] ** 2)
        output.append(en_sg / en_bg)

    return output


def find_a_in_b(a, b):
    nrows, ncols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [a.dtype]}

    c = np.intersect1d(a.view(dtype), b.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    c = c.view(a.dtype).reshape(-1, ncols)
    return c


def roc_build(im_filt, im_no_filt, im_bin, im_std, bound_sup=2, bound_inf=0):
    output = []
    output_en = []
    for i in range(0, im_filt.shape[2]):
        sx, sy = np.where(im_bin[:, :, i] == 1)
        bx, by = np.where(im_bin[:, :, i] == 0)
        sg_ef = []
        bg_ef = []
        energy = []
        for thr in np.linspace(bound_inf, bound_sup, 15):
            sxx, syy = np.where(im_filt[:, :, i] >= thr*im_std)
            s_pixels = find_a_in_b(np.stack((sxx, syy), axis=1),
                                   np.stack((sx, sy), axis=1))
            energy.append((im_no_filt[:, :, i][s_pixels[:, 0], s_pixels[:, 1]]).sum())
            im_thr = im_filt[:, :, i] >= thr*im_std
            sg_ef.append(sum(im_thr[sx, sy] == 1) / len(sx))
            bg_ef.append(sum(im_thr[bx, by] == 0) / len(bx))
        #for thr in np.linspace(bound_inf, bound_sup, bound_sup - bound_inf):
        #    im_thr = im_filt[:, :, i] >= thr
        #    sg_ef.append(sum(im_thr[sx, sy] == 1) / len(sx))
        #    bg_ef.append(sum(im_thr[bx, by] == 0) / len(bx))
        output.append([sg_ef, bg_ef])
        output_en.append(energy)
    return np.array(output), np.array(output_en)


def auc_calc(roc):
    for i in range(0, len(roc)):

        if i == 0:
            output = roc[0]
        else:
            output = np.append(output, roc[i], axis=0)
    sn_samples = output.shape[0]
    auc = []
    for i in range(0, sn_samples):
        auc.append(abs(sum(np.diff(output[i, 0, :]) * output[i, 1, :-1])))

    return auc


def im_rebin(im_input, rebin_factor=8):
    dim = im_input.shape
    n_imgs = dim[2]
    index = range(0, dim[0], rebin_factor)
    xx, yy, zz = np.meshgrid(index, index, range(0, n_imgs))
    return im_input[xx, yy, zz]


def get_energy(im_bin, im_desired):
    energy = []
    for i in range(0, im_bin.shape[2]):
        sx, sy = np.where(im_bin[:, :, i] == 1)  # signal pixels
        en = im_desired[sx, sy]
        energy.append(np.sum(en))
    return energy


class ResultGeneration:
    # constructor
    def __init__(self, folder, file_noise, run_number, batch_size, bound_sup, bound_inf, roi):
        self.folder = folder  # folder where there are h5 files
        self.file_noise = file_noise  # folder that has the noise files
        self.run_number = run_number  # the run number for the noise
        self.batch_size = batch_size  # the batch size number
        self.bound_sup = bound_sup    # the upper bound to remove outliers
        self.bound_inf = bound_inf    # the lower bound to remove outliers
        self.roi = np.sqrt(3)         # the cut value for roi
        # outputs
        self.filters_name = []
        self.windows = []
        self.snr = []
        self.energy = []
        self.roc_values = []
        self.energy_values = []
        self.energy_truth = []
        self.area = []

    # get pedestal from noise file
    def get_pedestal(self):
        fn = h5py.File(self.file_noise, 'r')  # read file noise
        ped = fn['mean']  # get pedestal
        std = fn['std']  # get std
        return ped[:, :, 820-self.run_number], std[:, :, 820-self.run_number]  # return values according to run number

    # get all h5 simulated files
    def get_file_names(self):
        return glob.glob(self.folder + '/*.h5')  # get all h5 files on interest folder

    def result2csv(self):
        self.snr = list(itertools.chain.from_iterable(self.snr))
        self.windows = list(itertools.chain.from_iterable(self.windows))
        self.filters_name = list(itertools.chain.from_iterable(self.filters_name))
        self.energy_truth = list(itertools.chain.from_iterable(self.energy_truth))
        self.area = list(itertools.chain.from_iterable(self.area))
        features = np.array([np.array(self.snr), self.windows, self.energy_truth, self.area]).T
        results = pd.DataFrame(np.append(features, np.array(self.filters_name).reshape(-1, 1), axis=1),
                               columns=['SNR', 'w_size', 'Energy', 'Area', 'filter'])
        results['SNR'] = (10*np.log(results['SNR'].astype('float')))
        results['window'] = results['w_size'].astype('float')
        results['Energy'] = results['Energy'].astype('float')
        results['Area'] = results['Area'].astype('float')
        results['ROCx'] = list(self.roc_values[:, 0, :])
        results['ROCy'] = list(self.roc_values[:, 1, :])
        results['Energy Estimated'] = list(self.energy_values)
        results.to_csv(r'../data/result.csv', index=None, header=True)

    # results for chosen metrics
    def calc_metrics(self, wrange, filters):
        full_files = self.get_file_names()
        ped, std = self.get_pedestal()
        std = np.expand_dims(std, axis=2)
        std = im_rebin(std, rebin_factor=4)
        std = std[:, :, 0]
        for file_name in full_files:
            f = h5py.File(file_name, 'r')
            print("Start Analysis")
            obj_x_train = f['x_train']
            obj_y_train = f['y_train']
            energy_factor = 1
            size = obj_x_train.shape
            im_dim = int(np.sqrt(size[1]))
            idx = list(range(0, size[0], self.batch_size))
            idy = list(range(self.batch_size, size[0], self.batch_size))
            idy.append(size[0])
            bar = Bar('Processing',  fill='-', max=len(wrange)*len(filters)*len(idx))
            count = 0
            for w in wrange:
                for fname in filters:
                    for i, j in zip(idx, idy):
                        a = time()
                        im_real = obj_x_train[i:j, :].T.reshape(im_dim, im_dim, j - i)
                        im_truth = obj_y_train[i:j, :].T.reshape(im_dim, im_dim, j - i)
                        alpha = energy_factor*np.ones(j-i)
                        #scale = im_truth.max(axis=0).max(axis=0)
                        # multiplica por alpha/pico -> alpha*I
                        im_truth = im_truth*alpha
                        # replicando valor do ruido para subtrair
                        multi_ped = np.repeat(ped[:, :, np.newaxis], im_real.shape[2], axis=2)
                        # rebinando
                        im_real = im_rebin(im_real, rebin_factor=4)
                        im_truth = im_rebin(im_truth, rebin_factor=4)
                        multi_ped = im_rebin(multi_ped, rebin_factor=4)
                        # removendo pedestal
                        im_no_pad = im_real - multi_ped
                        # saturando imagem
                        #im_no_pad[im_no_pad > self.bound_sup] = self.bound_sup
                        #im_no_pad[im_no_pad < self.bound_inf] = self.bound_inf
                        thresholds = 0  # vetor de thresholds para as imagens do batch
                        im_bin = im_truth > thresholds  # definindo como binaria a imagem maior que os threshold
                        # filtragem da imagem sem pedestal
                        im_filtered = linear_filtering(im_no_pad, w, fname)
                        # calculo do erro
                        self.snr.append(snr_calc(im_filtered, im_bin))
                        # calculo do erro sg-bg det
                        roc_curv, energy_curv = roc_build(im_filtered,
                                                          im_no_pad,
                                                          im_bin,
                                                          std,
                                                          self.bound_sup,
                                                          self.bound_inf)
                        if count == 0:
                            self.roc_values = roc_curv
                            self.energy_values = energy_curv
                        else:
                            self.roc_values = np.append(self.roc_values, roc_curv, axis=0)
                            self.energy_values = np.append(self.energy_values, energy_curv, axis=0)
                        #self.roc.append(roc_build(im_filtered, im_bin, self.bound_sup, self.bound_inf))
                        # armazenando contraste
                        #self.contrast.append(alpha)
                        # armazenando janelas
                        self.windows.append([w] * (j - i))
                        # armazenando nome dos filtros
                        self.filters_name.append([fname] * (j - i))
                        #armazenando energia
                        self.energy_truth.append(im_truth.sum(0).sum(0))
                        #self.en_real.append(get_energy(im_bin, im_truth))
                        #self.en_before.append(get_energy(im_bin, im_no_pad))
                        #self.en_after.append(get_energy(im_bin, im_filtered))
                        ## armazenando area
                        self.area.append(im_bin.sum(0).sum(0))
                        del roc_curv, energy_curv, im_real, im_truth, alpha, multi_ped, im_no_pad, im_bin, im_filtered
                        count += 1
                        bar.next()
                        b = time()-a
                        remaining = ((len(wrange) * len(filters) * len(idx)) - count) * b / 60
                        print('\n Estimated time per batch = ' + str(b))
                        print('\n Time remaining = ' + str(round(remaining))+' minutes')

            bar.finish()


def main():
    folder = '../data/Runs001'
    noise_folder = '../data/noise/noise_data.h5'
    run = 817
    batch_size = 32
    sup = 2
    inf = 0
    roi = 3
    data = ResultGeneration(folder, noise_folder, run, batch_size, sup, inf, roi)
    w_range = [1,3,5,7,9,11,13]
    filters = ['mean','gauss','median']
    data.calc_metrics(w_range, filters)
    data.result2csv()


if __name__ == "__main__":
    main()
