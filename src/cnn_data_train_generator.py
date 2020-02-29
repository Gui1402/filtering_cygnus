import numpy as np
import cv2
from settings import FilterSettings
import glob
import h5py
from hdf5_store import HDF5Store
import argparse

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--balanced', default=True, type=bool, help='Generate a balanced dataset')
args = parser.parse_args()


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


class CnnDataGen:

    def __init__(self, files_dir, file_noise, num_threads):
        self.file_dir = files_dir
        self.patch_size = 40
        self.stride = 10
        self.aug_times = 1
        self.batch_size = 8
        self.n_images = 80
        self.file_noise = file_noise
        self.num_threads = num_threads
        self.obj_x_train = []
        self.obj_y_train = []

    def get_train_files(self, idx_min, idx_max):
        file_names = glob.glob(self.file_dir + '/*.h5')
        file_name = file_names[0]
        f = h5py.File(file_name, 'r')
        obj_x_train = f['x_train']
        obj_y_train = f['y_train']
        self.obj_x_train = np.array(obj_x_train[idx_min:idx_max, :])
        self.obj_y_train = np.array(obj_y_train[idx_min:idx_max, :])

    def im_rebin(self, im_input, rebin_factor=8):
        """ rebin an input image by a rebin factor
            input (N,M,k)
            output (N//factor, M//factor, k)"""
        dim = im_input.shape
        index = range(0, dim[0], rebin_factor)
        try:
            n_imgs = dim[2]
            xx, yy, zz = np.meshgrid(index, index, range(0, n_imgs))
            im_rebined = im_input[xx, yy, zz]
        except IndexError:
            xx, yy = np.meshgrid(index, index)
            im_rebined = im_input[xx, yy]
        return im_rebined

    def get_pedestal(self, run_number):
        fn = h5py.File(self.file_noise, 'r')  # read file noise
        ped = fn['mean']  # get pedestal
        return ped[:, :, 820 - run_number]

    def gen_patches(self, index):
        img_in = self.obj_x_train[index, :]
        img_out = self.obj_y_train[index, :]
        dim = int(np.sqrt(len(img_out)))
        img_out = img_out.reshape(dim, dim) + 99.*np.ones((dim, dim))
        img_in = img_in.reshape(dim, dim)
        img_out = self.im_rebin(img_out, rebin_factor=1)
        img_in = self.im_rebin(img_in, rebin_factor=1)
        h, w = img_out.shape
        scales = [1, 0.9, 0.8, 0.7]
        patches_y = []
        patches_x = []
        log = []
        for s in scales:
            make_data = True
            count_pos = 0
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_out_scaled = cv2.resize(img_out, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            img_in_scaled = cv2.resize(img_in, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches

            for i in range(0, h_scaled - self.patch_size + 1, self.stride):
                for j in range(0, w_scaled - self.patch_size + 1, self.stride):
                    y = img_out_scaled[i:i + self.patch_size, j:j + self.patch_size]
                    if np.mean(y) == 99:
                        patch = False
                    else:
                        patch = True
                    x = img_in_scaled[i:i + self.patch_size, j:j + self.patch_size]
                    # data aug
                    for k in range(0, self.aug_times):
                        log.append(patch)
                        mode = np.random.randint(0, 8)
                        y_aug = data_aug(y, mode=mode)
                        x_aug = data_aug(x, mode=mode)
                        patches_y.append(y_aug)
                        patches_x.append(x_aug)

        return patches_y, patches_x, log

    def file_gen(self):
        count_data = 0
        idx = list(range(0, self.n_images, self.batch_size))
        idy = list(range(self.batch_size, self.n_images, self.batch_size))
        idy.append(self.n_images)
        shape = (self.patch_size, self.patch_size)
        store_x = HDF5Store('../data/train_x.h5', 'X', shape=shape)
        store_y = HDF5Store('../data/train_y.h5', 'Y', shape=shape)
        for id_min, id_max in zip(idx, idy):
            self.get_train_files(id_min, id_max)
            n_images = self.obj_x_train.shape[0]
            for i in range(0, n_images):
                patch_y, patch_x, log = self.gen_patches(i)
                if args.balanced:
                    ind_pos = np.where(np.array(log)==True)[0]
                    ind_neg = np.random.randint(low=0, high=len(log), size=len(ind_pos))
                    indexes = np.unique(np.append(ind_pos, ind_neg))
                else:
                    indexes = range(0, len(patch_x))
                for ind in indexes:
                    store_y.append(patch_y[ind])
                    store_x.append(patch_x[ind])
                    count_data += 1
                print('Picture ' + str(id_min + i) + ' to ' + str(id_max) + ' are finished...')
        print('Finished: '+str(count_data) + ' patches have been generated')


if __name__ == '__main__':
    settings = FilterSettings()
    dn_cnn = CnnDataGen(settings.data_folder, settings.noise_file, num_threads=16)
    dn_cnn.file_gen()







