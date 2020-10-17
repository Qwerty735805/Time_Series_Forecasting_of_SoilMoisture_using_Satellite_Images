import pandas as pd
import numpy as np
import skimage
from skimage.io import imread
from tqdm import tqdm
import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU, ReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam, SGD

def create_model(f,loss,activation):
    
    inputs = Input(shape=(256, 272, 1), name="inputs")
    x = inputs
    
    x = Conv2D(8, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((2,2))(x)

    x = Conv2D(16, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    skip_x = x

    x = Conv2D(16, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.add([x, skip_x])
    x = MaxPool2D((2,2))(x)

    x = Conv2D(32, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    skip_x1 = x

    x = Conv2D(32, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.add([x, skip_x1])
    x = MaxPool2D((2,2))(x)

    x = Conv2D(32, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((2,2))(x)

    x = Conv2D(16, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(8, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(4, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(2, (f,f), padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
    latent = Flatten()(x)
    
    encoder = Model(inputs, latent)
    
    
    encoded_input =  Input(shape=(544), name="encoded")
    x = Reshape((16,17,2))(encoded_input)
    

    x = Conv2DTranspose(4, (f,f),strides=1, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(8, (f,f),strides=1, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(16, (f,f),strides=1, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(32, (f,f),strides=2, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    skip_x2 = x

    x = Conv2DTranspose(32, (f,f),strides=1, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = tf.keras.layers.add([x, skip_x2])
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(16,(f,f), strides=2, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    skip_x3 = x

    x = Conv2DTranspose(16,(f,f), strides=1, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = tf.keras.layers.add([x, skip_x3])
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(8,(f,f), strides=2, padding="same",activation='elu')(x)
    x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(1,(f,f), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation, name="outputs")(x)
    if activation=='tanh':
        x = 0.5*x + 0.5 
    outputs = x
    
    decoder = Model(encoded_input, outputs)
    
    _inputs = Input(shape=(256, 272, 1), name="inputs")
    compressed = encoder(_inputs)
    reconstruction = decoder(compressed)
    
    autoencoder = Model(_inputs, reconstruction)
    autoencoder.compile(optimizer=Adam(1e-3), loss=loss)
    return autoencoder,encoder,decoder


s_ae3,s_e3,s_d3 =create_model(3,'binary_crossentropy','sigmoid')
r_ae3,r_e3,r_d3 =create_model(3,tf.keras.losses.MeanAbsoluteError(),'relu')
t_ae3,t_e3,t_d3 =create_model(5,tf.keras.losses.MeanAbsoluteError(),'tanh')


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 2e-3 * 0.95 ** x)

history_s_ae3 = s_ae3.fit(train_data,train_data,epochs=50,batch_size=32,shuffle=False,
    validation_data=(valid_data,valid_data),verbose=2,steps_per_epoch=len(train_data) // 32,callbacks=[reduce_lr])
history_r_ae3= r_ae3.fit(train_data,train_data,epochs=50,batch_size=32,shuffle=False,
    validation_data=(valid_data,valid_data),verbose=2,steps_per_epoch=len(train_data) // 32,callbacks=[reduce_lr])
history_t_ae3= t_ae3.fit(train_data,train_data,epochs=50,batch_size=32,shuffle=False,
        validation_data=(valid_data,valid_data),verbose=2,steps_per_epoch=len(train_data) // 32,callbacks=[reduce_lr])

s_d3.save("Models/DR/s_dec.h5")
s_e3.save("Models/DR/s_enc.h5")
s_ae3.save("Models/DR/s.h5")

r_d3.save("Models/DR/r_dec.h5")
r_e3.save("Models/DR/r_enc.h5")
r_ae3.save("Models/DR/r.h5")

t_d3.save("Models/DR/t_dec.h5")
t_e3.save("Models/DR/t_enc.h5")
t_ae3.save("Models/DR/t.h5")