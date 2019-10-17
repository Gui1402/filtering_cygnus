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



class DataGenerator():
    def __init__(self, file_name, batch_size, data_split=10):
        self.hf = h5py.File(file_name, 'r')
        y_all = self.hf['y_train'][:]
        self.total_len = len(y_all)
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
            yield np.reshape(batch_x,(self.batch_size,512,512,1)), np.reshape(batch_y,(self.batch_size,512,512,1))


#plots = False
#
#def plotImgs(indexes,imgReal,imgTruth):
#    fig,axes       = plt.subplots(2,len(indexes))
#    for i,imNum in list(enumerate(indexes)):
#        img        = imgReal[imNum,:].reshape(512,512)
#        imgt       = imgTruth[imNum,:].reshape(512,512)
#        ax         = axes[:,i]
#        ax[1].imshow(imgt,cmap = 'gray')
#        ax[0].set_title(r'$\alpha$ = ' + str(alpha[imNum]))
#        ax[1].set_title('Truth')
#        
#        xh,yh      = np.where(np.diff(imgt,axis = 0)!=0)
#        xv,yv      = np.where(np.diff(imgt,axis = 1)!=0)
#        aux        = img
#        aux[xh,yh] = 0
#        aux[xv,yv] = 0
#        ax[0].imshow(aux,cmap = 'gray',vmax = 135,vmin = 85)
        
        
def networkModel():
    # Network parameters
    input_shape = (512, 512, 1)
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
        
        
def denoisingNNTraining(file_name,batch_size):
    
    model  = networkModel()
    training_generator = DataGenerator(file_name, batch_size=batch_size).generate()
    model.fit_generator(generator=training_generator, 
                    epochs=1,
                    steps_per_epoch=20, workers=4, 
                    use_multiprocessing=True, 
                    verbose=1)
    
    
    

       
file_name  = "test.h5"
net        = denoisingNNTraining(file_name,batch_size = 10)





