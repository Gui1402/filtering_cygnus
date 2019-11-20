import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
import h5py
import glob
import itertools
from progress.bar import Bar



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
    else:
        print("filter has not found")
        # se nenhum filtro for escolhido convolui com um impulso
        mask[1, 1, :] = 1
    mask[:, :, 0] = np.rot90(np.rot90(mask[:, :, 0]))
    # convolui com filtro rotacionado de 180 graus
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


def roc_build(im_filt, im_bin, bound_sup, bound_inf):
    output = []
    for i in range(0, im_filt.shape[2]):
        sx, sy = np.where(im_bin[:, :, i] == 1)
        bx, by = np.where(im_bin[:, :, i] == 0)
        sg_ef = []
        bg_ef = []
        for thr in np.linspace(bound_inf, bound_sup, bound_sup - bound_inf):
            im_thr = im_filt[:, :, i] >= thr
            sg_ef.append(sum(im_thr[sx, sy] == 1) / len(sx))
            bg_ef.append(sum(im_thr[bx, by] == 0) / len(bx))
        output.append([sg_ef, bg_ef])
    return np.array(output)


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
        self.roc = []
        self.contrast = []
        self.energy = []

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
        self.contrast = list(itertools.chain.from_iterable(self.contrast))
        self.windows = list(itertools.chain.from_iterable(self.windows))
        self.filters_name = list(itertools.chain.from_iterable(self.filters_name))
        self.energy = list(itertools.chain.from_iterable(self.energy))
        features = np.array([self.contrast, np.array(self.snr), auc_calc(self.roc), self.windows, self.energy]).T
        results = pd.DataFrame(np.append(features, np.array(self.filters_name).reshape(-1, 1), axis=1),
                               columns=['contrast', 'SNR', 'AUC', 'w_size', 'Energy', 'filter'])
        results['contrast'] = results['contrast'].astype('float32')
        results['AUC'] = results['AUC'].astype('float32')
        results['SNR'] = (10*np.log(results['SNR'].astype('float32')))
        results['window'] = results['w_size'].astype('float')
        results['Energy'] = results['Energy'].astype('float32')
        results.to_csv(r'../data/result_1.csv', index=None, header=True)

    # results for chosen metrics
    def calc_metrics(self, wrange, filters):
        full_files = self.get_file_names()
        for file_name in full_files:
            f = h5py.File(file_name, 'r')
            print("Start Analysis")
            obj_x_train = f['x_train']
            obj_y_train = f['y_train']
            energy_factor = 1
            size = obj_x_train.shape
            im_dim = int(np.sqrt(size[1]))
            ped, _ = self.get_pedestal()
            idx = list(range(0, size[0], self.batch_size))
            idy = list(range(self.batch_size, size[0], self.batch_size))
            idy.append(size[0])
            bar = Bar('Processing',  fill='-', max=len(wrange)*len(filters)*len(idx))
            for w in wrange:
                for fname in filters:
                    count = 0
                    for i, j in zip(idx, idy):
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
                        self.roc.append(roc_build(im_filtered, im_bin, self.bound_sup, self.bound_inf))
                        # armazenando contraste
                        self.contrast.append(alpha)
                        # armazenando janelas
                        self.windows.append([w] * (j - i))
                        # armazenando nome dos filtros
                        self.filters_name.append([fname] * (j - i))
                        #armazenando energia
                        self.energy.append((im_truth.sum(0).sum(0))/(im_bin.sum(0).sum(0)))
                        count += 1
                        bar.next()
            bar.finish()


def main():
    folder = '../data/Runs001'
    noise_folder = '../data/noise/noise_data.h5'
    run = 818
    batch_size = 32
    sup = 111
    inf = -10
    roi = 3
    data = ResultGeneration(folder, noise_folder, run, batch_size, sup, inf, roi)
    w_range = range(1, 7, 2)
    filters = ['mean', 'gauss']
    data.calc_metrics(w_range, filters)
    data.result2csv()


if __name__ == "__main__":
    main()
