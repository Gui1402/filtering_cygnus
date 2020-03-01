from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import numpy as np
import glob
import h5py


class DataGenerator:
    def __init__(self, file_name, batch_size, data_split=100):
        self.hf = h5py.File(file_name, 'r')
        y_all = self.hf['y_train'].shape[0]
        self.total_len = y_all
        self.batch_size = batch_size
        self.idx = 0
        self.len_segment = int(self.total_len / data_split)
        self.cur_seg_idx = 0
        self.x_cur = self.hf['x_train'][:self.len_segment]
        self.y_cur = self.hf['y_train'][:self.len_segment]

    def next_seg(self):
        self.cur_seg_idx += self.len_segment
        self.x_cur = self.hf['x_train'][self.cur_seg_idx:self.cur_seg_idx+self.len_segment]
        self.y_cur = self.hf['y_train'][self.cur_seg_idx:self.cur_seg_idx+self.len_segment]
        
    def generate(self):
        while 1:
            idx = self.idx
            if idx >= self.len_segment:
                self.next_seg()
                idx = 0
            
            if idx + self.batch_size >= self.len_segment:
                batch_x = self.x_cur[idx:]
                batch_y = self.y_cur[idx:]
            else:
                batch_x = self.x_cur[idx:(idx + self.batch_size)]
                batch_y = self.y_cur[idx:(idx + self.batch_size)]
            self.idx = idx + self.batch_size
            im_size = int(np.sqrt(batch_y.shape[1]))
            im_x = np.reshape(batch_x, (batch_x.shape[0], im_size, im_size, 1))
            im_y = np.reshape(batch_y, (batch_y.shape[0], im_size, im_size, 1))
            yield im_x, im_y

class AutoEnc:

    def __init__(self, f_name, batch_size):
        self.file_name = f_name  # name of file
        self.batch_size = batch_size  # size of batch
        with h5py.File(self.file_name, 'r') as hf:  # read file
            shape = hf['y_train'].shape  # get data shape
        train_len = shape[0] # get number of img
        self.x_len = int(train_len / self.batch_size)  # number of batches
        self.im_shape = int(np.sqrt(shape[1]))  # shape of images
        # output
        self.auto_encoder = []

    def network_model(self, kernel_size, latent_dim, layer_filters):
        input_shape = (self.im_shape, self.im_shape, 1)  # input of our network
        inputs = Input(shape=input_shape, name='encoder_input')  # config input encoder
        x = inputs
        # creating network
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)
        # Shape info needed to build Decoder Model
        shape = K.int_shape(x)
        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)
        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
        # Build the Decoder Model
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        self.auto_encoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        self.auto_encoder.summary()
        self.auto_encoder.compile(loss='mse', optimizer='adam')

    def network_train(self, n_epochs):
        training_generator = DataGenerator(self.file_name, self.batch_size).generate()
        self.auto_encoder.fit_generator(generator=training_generator,
                                        epochs=n_epochs,
                                        steps_per_epoch=self.x_len,
                                        workers=1,
                                        use_multiprocessing=False,
                                        verbose=1)


def networkModel():
    # Network parameters
    input_shape = (256, 256, 1)
    #batch_size = 3
    kernel_size = 3
    latent_dim = 16
    # Encoder/Decoder number of CNN layers and filters per layer
    layer_filters = [32, 64]
    
    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Stack of Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use MaxPooling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)
    
    
    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)
    
    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    
    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()
    
    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    
    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)
    
    x = Conv2DTranspose(filters=1,
                        kernel_size=kernel_size,
                        padding='same')(x)
    
    outputs = Activation('sigmoid', name='decoder_output')(x)
    
    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    
    autoencoder.compile(loss='mse', optimizer='adam')
    return autoencoder
        
        
def denoisingNNTraining(file_name, batch_size, x_len):
    
    model  = networkModel()
    training_generator = DataGenerator(file_name, batch_size=batch_size).generate()
    model.fit_generator(generator=training_generator, 
                        epochs=1,
                        steps_per_epoch=x_len,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=1)
    return model


def main():
    # file with data
    file_name = '../Run_2019-11-07.h5'
    # batch size
    batch_size = 30
    # creating obj auto_encoder
    auto_encoder_denoising = AutoEnc(f_name=file_name, batch_size=batch_size)
    # network parameters
    kernel_size = 3
    latent_dim = 16
    layer_filters = [32, 64]
    # creating network model
    auto_encoder_denoising.network_model(kernel_size=kernel_size, latent_dim=latent_dim, layer_filters=layer_filters)
    # training network
    auto_encoder_denoising.network_train(n_epochs=1)


    #
    #with h5py.File(file_name, 'r') as hf:
    #    shape = hf['y_train'].shape
    #train_len = shape[0]
    #batch_size = 30
    #x_len = int(train_len/batch_size)
    #net = denoisingNNTraining(file_name, batch_size=batch_size, x_len=x_len)
    #return net


if __name__ == "__main__":
    main()




