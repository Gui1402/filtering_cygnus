import sys
sys.path.append('/fc_aide')
sys.path.append('fc_aide/')
from core.test_blind_ft import Fine_tuning as test_ft
from scipy import misc
import numpy as np
import h5py
import matplotlib.pyplot as plt


f = h5py.File('../data/Runs001/first_simulations.h5', 'r')
index = 0
noisy_image = f['x_train'][index, :].reshape(2048, 2048)
clean_image = f['y_train'][index, :].reshape(2048, 2048)
noisy_image = np.clip(noisy_image, 76, 116)
xx, yy = np.meshgrid(range(0, 2048, 4), range(0, 2048, 4))
noisy_image = noisy_image[xx, yy]
clean_image = clean_image[xx, yy]
clean_image = clean_image*255/clean_image.max()

t_ft = test_ft(clean_image, noisy_image, 2)
denoised_img = t_ft.fine_tuning()
plt.imsave('denoised.jpg', denoised_img)

