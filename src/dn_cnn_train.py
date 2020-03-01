# -*- coding: utf-8 -*-

import argparse
import logging
import os, time
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
import models
import h5py
from settings import FilterSettings

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--run', default=817, type=int, help='run number')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
args = parser.parse_args()


save_dir = '../data/snapshot/save_' + args.model + '_' + 'run' + str(args.run) + '_' + time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    # log
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y %H:%M:%S',
                        filename=save_dir + 'info.log',
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
        lr = initial_lr / 10
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
    settings = FilterSettings()
    while (True):
        np.random.shuffle(indices)  # shuffle
        for i in range(0, len(indices), batch_size):
            index_list = indices[i:i + batch_size]
            ge_batch_x, ge_batch_y = get_h5_from_slices(x_, y_, index_list)
            up_bound = 99 + settings.sup
            low_bound = 99 - settings.inf
            den = up_bound-low_bound
            ge_batch_x = np.clip(ge_batch_x, low_bound, up_bound)
            ge_batch_y = np.clip(ge_batch_y, low_bound, up_bound)
            ge_batch_x = (ge_batch_x - low_bound) / den
            ge_batch_y = (ge_batch_y - low_bound) / den
            yield ge_batch_x, ge_batch_y


def train():
    data_x, data_y = load_train_data()
    if args.pretrain:
        model = load_model(args.pretrain, compile=False)
    else:
        if args.model == 'DnCNN':
            model = models.DnCNN()
        elif args.model == 'BRDNet':
            model = models.BRDNet()
        else:
            print('Invalid model')
    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'])
    # use call back functions
    ckpt = ModelCheckpoint(save_dir + '/model_{epoch:02d}.h5', monitor='val_loss',
                           verbose=0, period=args.save_every)
    csv_logger = CSVLogger(save_dir + '/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # train
    history = model.fit_generator(train_datagen(data_x, data_y, batch_size=args.batch_size),
                                  steps_per_epoch=len(data_x) // args.batch_size,
                                  epochs=args.epoch,
                                  verbose=1,
                                  callbacks=[ckpt, csv_logger, lr])

    return model


if __name__ == '__main__':
    model = train()
