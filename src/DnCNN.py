import numpy as np
import cv2
from multiprocessing import Pool
from settings import FilterSettings
from filter_analysis import ResultGeneration
import glob
import h5py
from hdf5_store import HDF5Store
import tables







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


class DnCNN:
    def __init__(self, files_dir, file_noise, num_threads):
        self.file_dir = files_dir
        self.patch_size = 40
        self.stride = 10
        self.aug_times = 1
        self.batch_size = 32
        self.n_images = 100
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
            n_imgs = 1
            xx, yy = np.meshgrid(index, index)
            im_rebined = im_input[xx, yy]
        return im_rebined

    def get_pedestal(self, run_number):
        fn = h5py.File(self.file_noise, 'r')  # read file noise
        ped = fn['mean']  # get pedestal
        return ped[:, :, 820 - run_number]

    def gen_patches(self, index):
        # read image
        img_in = self.obj_x_train[index, :]
        img_out = self.obj_y_train[index, :]
        dim = int(np.sqrt(len(img_out)))
        img_out = img_out.reshape(dim, dim) + 99.*np.ones((dim, dim))
        img_in = img_in.reshape(dim, dim)
        img_out = self.im_rebin(img_out, rebin_factor=4)
        img_in = self.im_rebin(img_in, rebin_factor=4)
        h, w = img_out.shape
        scales = [1, 0.9, 0.8, 0.7]
        patches_y = []
        patches_x = []

        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_out_scaled = cv2.resize(img_out, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            img_in_scaled = cv2.resize(img_in, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled - self.patch_size + 1, self.stride):
                for j in range(0, w_scaled - self.patch_size + 1, self.stride):
                    y = img_out_scaled[i:i + self.patch_size, j:j + self.patch_size]
                    x = img_in_scaled[i:i + self.patch_size, j:j + self.patch_size]
                    # data aug
                    for k in range(0, self.aug_times):
                        mode = np.random.randint(0, 8)
                        y_aug = data_aug(y, mode=mode)
                        x_aug = data_aug(x, mode=mode)
                        patches_y.append(y_aug)
                        patches_x.append(x_aug)

        return patches_y, patches_x

    def file_gen(self):
        idx = list(range(0, self.n_images, self.batch_size))
        idy = list(range(self.batch_size, self.n_images, self.batch_size))
        idy.append(self.n_images)
        shape = (self.patch_size, self.patch_size)
        store_x = HDF5Store('../data/train_x.h5', 'X', shape=shape)
        store_y = HDF5Store('../data/train_y.h5', 'Y', shape=shape)
        for id_min, id_max in zip(idx, idy):
            self.get_train_files(id_min, id_max)
            #res = []
            n_images = self.obj_x_train.shape[0]
            for i in range(0, n_images):
                # use multi-process to speed up
                #p = Pool(self.num_threads)
                #patch = p.map(self.gen_patches, range(i, min(i + self.num_threads, n_images)))
                # patch = p.map(gen_patches,file_list[i:i+num_threads])
                patch_y, patch_x = self.gen_patches(i)
                for x, y in zip(patch_x, patch_y):
                    store_y.append(y)
                    store_x.append(x)
                    #res += x
                    #print(x.shape)

                print('Picture ' + str(id_min + i) + ' to ' + str(id_max) + ' are finished...')
        #res = np.array(res)
        #return res


if __name__ == '__main__':
    settings = FilterSettings()
    dn_cnn = DnCNN(settings.data_folder, settings.noise_file, num_threads=16)
    dn_cnn.file_gen()







