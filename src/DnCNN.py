import numpy as np
import cv2
from multiprocessing import Pool
from settings import FilterSettings
import glob
import h5py
import copy



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
        self.obj_x_train = []
        self.obj_y_train = []

    def get_train_files(self):
        file_names = glob.glob(self.file_dir + '/*.h5')
        file_name = file_names[0]
        f = h5py.File(file_name, 'r')
        self.obj_x_train = f['x_train']
        self.obj_y_train = f['y_train']

    def hf5_array(self, idx_min, idx_max):
        return np.array(self.obj_y_train[idx_min:idx_max, :])


    def gen_patches(self, index):
        # read image

        img = self.obj_y_train[index, :]
        dim = int(np.sqrt(len(img)))
        img = img.reshape(dim, dim)
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
        self.get_train_files()
        batch_size = 32



        size = self.obj_x_train.shape
        n_images = size[0]
        res = []
        for i in range(0, n_images, self.num_threads):
            # use multi-process to speed up
            p = Pool(self.num_threads)
            patch = p.map(self.gen_patches, range(i, min(i + self.num_threads, n_images)))
            # patch = p.map(gen_patches,file_list[i:i+num_threads])
            for x in patch:
                res += x
        res = np.array(res, dtype='uint8')
        return res


if __name__ == '__main__':
    settings = FilterSettings()
    dn_cnn = DnCNN(settings.data_folder, num_threads=16)
    dn_cnn.file_gen()
