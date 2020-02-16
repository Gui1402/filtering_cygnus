import numpy as np
import cv2
from multiprocessing import Pool
from settings import FilterSettings
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
    def __init__(self, files_dir, num_threads):
        self.file_dir = files_dir
        self.patch_size = 40
        self.stride = 10
        self.aug_times = 1
        self.batch_size = 32
        self.n_images = 100
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

    def gen_patches(self, index):
        # read image

        img = self.obj_y_train[index, :]
        dim = int(np.sqrt(len(img)))
        img = img.reshape(dim, dim)
        img = self.im_rebin(img, rebin_factor=16)
        h, w = img.shape
        scales = [1, 0.9, 0.8, 0.7]
        patches = []

        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled - self.patch_size + 1, self.stride):
                for j in range(0, w_scaled - self.patch_size + 1, self.stride):
                    x = img_scaled[i:i + self.patch_size, j:j + self.patch_size]
                    # data aug
                    for k in range(0, self.aug_times):
                        x_aug = data_aug(x, mode=np.random.randint(0, 8))
                        patches.append(x_aug)

        return patches

    def file_gen(self):
        idx = list(range(0, self.n_images, self.batch_size))
        idy = list(range(self.batch_size, self.n_images, self.batch_size))
        idy.append(self.n_images)
        shape = (self.patch_size, self.patch_size)
        hdf5_store = HDF5Store('../data/train.h5', 'X', shape=shape)
        for id_min, id_max in zip(idx, idy):
            self.get_train_files(id_min, id_max)
            #res = []
            n_images = self.obj_x_train.shape[0]
            for i in range(0, n_images):
                # use multi-process to speed up
                #p = Pool(self.num_threads)
                #patch = p.map(self.gen_patches, range(i, min(i + self.num_threads, n_images)))
                # patch = p.map(gen_patches,file_list[i:i+num_threads])
                patch = self.gen_patches(i)
                for x in patch:
                    hdf5_store.append(x)
                    #res += x
                    #print(x.shape)

                print('Picture ' + str(i) + ' to ' + str(i + self.num_threads) + ' are finished...')
        #res = np.array(res)
        #return res


if __name__ == '__main__':
    settings = FilterSettings()
    dn_cnn = DnCNN(settings.data_folder, num_threads=16)
    dn_cnn.file_gen()







