import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy import signal
import h5py
import glob
import itertools


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


def calc_erro(imgTruth, imgFiltered, metric='PSNR'):
    mse = ((imgTruth - imgFiltered) ** 2).sum(axis=0).sum(axis=0)
    if metric == 'PSNR':
        return 20 * np.log10(135 - 99 + 10) - 10 * np.log10(mse)
    else:
        # print('Invalid metric, returning mse')
        return mse


def matrix3scale(image, flag):
    num = (image - image.min(axis=0).min(axis=0))
    if (flag == 1):
        sup = 35  # caso positivo considerar imagem truth
    else:
        sup = image.max(axis=0).max(axis=0)  # caso negativo imagem real
    den = (sup - image.min(axis=0).min(axis=0))
    return num / den


def snrcalc(im_filtrada, im_bin):
    output = []
    truth_spx, truth_spy, truth_spz = np.where(im_bin == True)  # pixels pertencentes a ROI definida
    truth_bpx, truth_bpy, truth_bpz = np.where(im_bin == False)  # pixels fora da ROI definida
    for index in np.unique(truth_spz):
        isx, isy = truth_spx[truth_spz == index], truth_spy[truth_spz == index]
        ibx, iby = truth_bpx[truth_bpz == index], truth_bpy[truth_bpz == index]
        en_sg = sum(im_filtrada[isx, isy, index] ** 2)
        en_bg = sum(im_filtrada[ibx, iby, index] ** 2)
        output.append(en_sg / en_bg)
    return output


def sgbgcalc(im_filtrada, im_bin):
    boundsup = 135 - 99 + 10
    boundinf = 85 - 99 - 10
    output = []
    for i in range(0, im_filtrada.shape[2]):
        sx, sy = np.where(im_bin[:, :, i] == 1)
        bx, by = np.where(im_bin[:, :, i] == 0)
        sg_ef = []
        bg_ef = []
        for thr in np.linspace(boundinf, boundsup, boundsup - boundinf):
            im_thr = im_filtrada[:, :, i] >= thr
            sg_ef.append(sum(im_thr[sx, sy] == 1) / len(sx))
            bg_ef.append(sum(im_thr[bx, by] == 0) / len(bx))
        output.append([sg_ef, bg_ef])
    return np.array(output)


folder = '../data/Runs001'
# lista com todos os arquivos .h5
fullfiles = glob.glob(folder + '/*.h5')
# leitura dos arquivos de ruido
file_noise = '../data/noise/noise_data.h5'
fn = h5py.File(file_noise, 'r')
ped = fn['mean']
# run 817
ped = ped[:, :, 1]

# Leitura dos dados
# tamanho dos pacotes de imgs processados
batch_size = 32
# limitantes superiores e inferiores
boundsup = 135 - 99 + 10
boundinf = 85 - 99 - 10
# outputs
result_dt = []
erro = []
contraste = []
wvalue = []
filters_name = []
n_sigma = np.sqrt(3)  # quantos sigmas pegaremos do sinal
for filename in fullfiles:
    # leitura do h5
    f = h5py.File(filename, 'r')
    # matriz com imagens ruidosas
    objxtrain = f['x_train']
    # matriz com imagens truth correspondentes
    objytrain = f['y_train']
    # vetor com intensidade correspondente
    objalpha = f['alpha']
    # tamanho das matrizes #imagens x MxN
    size = objxtrain.shape
    # dimensao da imagem quadrada por default
    imdim = int(np.sqrt(size[1]))
    # adquirindo imagem pedestal centrada na origem
    p = ped[1024 - imdim // 2:1024 + imdim // 2, 1024 - imdim // 2:1024 + imdim // 2]
    # vetores de varredura dos batches
    idx = list(range(0, size[0], batch_size))
    idy = list(range(batch_size, size[0], batch_size))
    idy.append(size[0])
    wrange = range(1, 7, 2)
    filters = ['mean', 'gauss']
    result_dt = []
    for fname in filters:
        for w in wrange:
            count = 0
            for i, j in zip(idx, idy):
                # reshaping para transformar batches de imagens em dim x dim x batch_size
                imReal = objxtrain[i:j, :].T.reshape(imdim, imdim, j - i)
                imTruth = objytrain[i:j, :].T.reshape(imdim, imdim, j - i)
                alpha = objalpha[i:j]
                # normalizar pico para 1
                scale = imTruth.max(axis=0).max(axis=0)
                # multiplica por alpha/pico -> alpha*I
                imTruth = imTruth * (alpha / scale)
                # replicando valor do ruido para subtrair
                multiPed = np.repeat(p[:, :, np.newaxis], imReal.shape[2], axis=2)
                # removendo pedestal
                imNoPad = imReal - multiPed
                # saturando imagem
                imNoPad[imNoPad > boundsup] = boundsup
                imNoPad[imNoPad < boundinf] = boundinf
                thresholds = alpha * (np.exp(-n_sigma ** 2))  # vetor de thresholds para as imagens do batch
                im_bin = imTruth >= thresholds  # definindo como binaria a imagem maior que os threshold

                # filtragem da imagem sem pedestal
                imFiltered = linear_filtering(imNoPad, w, fname)
                # calculo do erro
                erro.append(snrcalc(imFiltered, im_bin))
                # calculo do erro sg-bg det
                result_dt.append(sgbgcalc(imFiltered, im_bin))
                # armazenando contraste
                contraste.append(alpha)
                # armazenando janelas
                wvalue.append([w] * (j - i))
                # armazenando nome dos filtros
                filters_name.append([fname] * (j - i))
                count += 1
                print('Batch' + str(count) + ' done' + ' filter --> ' + fname + str(w))

# convertendo lista de listas em arrays 1D
erro = list(itertools.chain.from_iterable(erro))
contraste = list(itertools.chain.from_iterable(contraste))
wvalue = np.array(list(itertools.chain.from_iterable(wvalue)))
filters_name = np.array(list(itertools.chain.from_iterable(filters_name)))

# calculando auc
for i in range(0, len(result_dt)):
    if (i == 0):
        output = result_dt[0]
    else:
        output = np.append(output, result_dt[i], axis=0)
sn_samples = output.shape[0]
auc = []
for i in range(0, sn_samples):
    auc.append(abs(sum(np.diff(output[i, 0, :]) * output[i, 1, :-1])))

# concatenando arrays 1D em 1 array 3 para analise
result_en = np.array([contraste, 10 * np.log(erro), auc, wvalue]).T

results = pd.DataFrame(np.append(result_en, filters_name.reshape(-1, 1), axis=1),
                       columns=['contrast', 'erro', 'auc', 'window', 'filter'])
results['contrast'] = results['contrast'].astype('float32')
results['auc'] = results['auc'].astype('float32')
results['erro'] = results['erro'].astype('float32')
results['window'] = results['window'].astype('float')
export_csv = results.to_csv(r'../data/export_dataframe2.csv', index=None,
                            header=True)  # Don't forget to add '.csv' at the end of the path
