# -*- coding: utf-8 -*-

import argparse
import logging
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
from skimage.measure import compare_psnr, compare_ssim
import models
import h5py

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='./data/npy_data/clean_patches.npy', type=str, help='path of train data')
parser.add_argument('--test_dir', default='./data/Test/Set68', type=str, help='directory of test dataset')
parser.add_argument('--run', default=817, type=int, help='run number')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
args = parser.parse_args()

if not args.only_test:
    save_dir = '../data/snapshot/save_' + args.model + '_' + 'run' + str(args.run) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # log
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename=save_dir+'info.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(args)
    
else:
    save_dir = '/'.join(args.pretrain.split('/')[:-1]) + '/'

      
def load_train_data():
    x_train = h5py.File('../data/train_x.h5', 'r')['X']
    y_train = h5py.File('../data/train_y.h5', 'r')['Y']
    logging.info('loading train data...')
    return x_train, y_train


def step_decay(epoch):
    initial_lr = args.lr
    if epoch < 50:
        lr = initial_lr
    else:
        lr = initial_lr/10
    return lr


def get_h5_from_slices(obj_x, obj_y, indexes):

    output_array_x = np.empty(shape=(0, obj_x.shape[1], obj_x.shape[2], 1))
    output_array_y = np.empty(shape=(0, obj_y.shape[1], obj_y.shape[2], 1))
    for i in indexes:
        output_array_x = np.append(output_array_x, obj_x[i, ...].reshape(1, obj_x.shape[1], obj_x.shape[2], 1), axis=0)
        output_array_y = np.append(output_array_y, obj_y[i, ...].reshape(1, obj_y.shape[1], obj_y.shape[2], 1), axis=0)
    return output_array_x, output_array_y


def train_datagen(x_, y_, batch_size=8):
    # y_ is the tensor of clean patches
    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        for i in range(0, len(indices), batch_size):
            index_list = indices[i:i+batch_size]
            ge_batch_x, ge_batch_y = get_h5_from_slices(x_, y_, index_list)
            ge_batch_x = np.clip(ge_batch_x, 74, 116)
            ge_batch_y = np.clip(ge_batch_y, 74, 116)
            ge_batch_x = (ge_batch_x-74)/(116-74)
            ge_batch_y = (ge_batch_y - 74) / (116 - 74)
            yield ge_batch_x, ge_batch_y
        

def train():
    data_x, data_y = load_train_data()
    if args.pretrain:
        model = load_model(args.pretrain, compile=False)
    else:   
        if args.model == 'DnCNN':
            model = models.DnCNN()
    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'])
    # use call back functions
    ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss', 
                           verbose=0, period=args.save_every)
    csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # train 
    history = model.fit_generator(train_datagen(data_x, data_y, batch_size=args.batch_size),
                                  steps_per_epoch=len(data_x)//args.batch_size,
                                  epochs=args.epoch,
                                  verbose=1,
                                  callbacks=[ckpt, csv_logger, lr])
    
    return model


def test(model):
    f = h5py.File('../data/Runs001/first_simulations.h5', 'r')
    print("Start Analysis")
    obj_x_train = f['x_train']
    
    #print('Start to test on {}'.format(args.test_dir))
    #out_dir = save_dir + args.test_dir.split('/')[-1] + '/'
    #if not os.path.exists(out_dir):
    #        os.mkdir(out_dir)
            
    #name = []
    #psnr = []
    #ssim = []
    #file_list = glob.glob('{}/*.png'.format(args.test_dir))
    # for file in file_list:
    #     # read image
    #     img_clean = np.array(Image.open(file), dtype='float32') / 255.0
    #     img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)
    #     img_test = img_test.astype('float32')
    #     img_test = np.clip(obj_x_train[85, :], 74, 116)
    #     img_test = (img_test - 74) / (116 - 74)
    #     img_test = img_test.reshape(2048, 2048)
    #     # predict
    #     x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1)
    #     y_predict = model.predict(x_test)
    #     # calculate numeric metrics
    #     img_out = y_predict.reshape(img_clean.shape)
    #     img_out = np.clip(img_out, 0, 1)
    #     psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
    #     ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
    #     psnr.append(psnr_denoised)
    #     ssim.append(ssim_denoised)
    #     # save images
    #     filename = file.split('/')[-1].split('.')[0]    # get the name of image file
    #     name.append(filename)
    #     img_test = Image.fromarray((img_test*255).astype('uint8'))
    #     img_test.save(out_dir+filename+'_sigma'+'{}_psnr{:.2f}.png'.format(args.sigma, psnr_noise))
    #     img_out = Image.fromarray((img_out*255).astype('uint8'))
    #     img_out.save(out_dir+filename+'_psnr{:.2f}.png'.format(psnr_denoised))
    #
    # psnr_avg = sum(psnr)/len(psnr)
    # ssim_avg = sum(ssim)/len(ssim)
    # name.append('Average')
    # psnr.append(psnr_avg)
    # ssim.append(ssim_avg)
    # print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    #
    # pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 'ssim':np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)


if __name__ == '__main__':
    if args.only_test:
        model = load_model(args.pretrain, compile=False)
        test(model)
    else:
        model = train()
        #test(model)
