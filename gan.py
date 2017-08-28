import argparse
import os
import numpy as np
from PIL import Image
import math
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers import Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam
import mnist_reader
import pandas as pd 
import sys

BATCH_SIZE=32
NUM_EPOCH = 50
Generator_file='data/Generator/'
EPOCH_Count=100

def generator(input_dim=100,units=1024,activation='relu'):
    model=Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7,7,128),input_shape=(128*7*7,)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1,(5,5),padding='same'))
    model.add(Activation('tanh'))
    return model

def discrminator(input_shape=(28,28,1),np_filter=64):
    model=Sequential()
    model.add(Conv2D(np_filter,(5,5),strides=(2,2),padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*np_filter,(5,5),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4*np_filter))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def show_progress(e,i,g0,d0,g1,d1):
    sys.stdout.write("\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (e,i,g0,d0,g1,d1))
    sys.stdout.flush()

def combine_images(generated_images):
    total,width,height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

def train():
    #(X_train,y_train),(_,_)=mnist.load_data()
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_train=(X_train.astype(np.float32)-127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

    g=generator()
    d=discrminator()

    opt=Adam(lr=0.0002,beta_1=0.5)
    d.trainable=True
    d.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
    d.trainable=False
    dcgan=Sequential([g,d])
    dcgan.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
    #BATCH_SIZE a times
    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    z_pred=np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_g=[1]*BATCH_SIZE
    y_d_true=[1]*BATCH_SIZE
    y_d_gan=[0]*BATCH_SIZE

    for epoch in range(1,EPOCH_Count):
        for index in range(num_batches):
            #initial
            X_d_true=X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            X_g=np.array([np.random.normal(0,0.5,100)for _ in range(BATCH_SIZE)])
            X_d_gan=g.predict(X_g,verbose=0)

            #test discriminator
            #positive
            d_loss=d.train_on_batch(X_d_true,y_d_true)
            #negative
            d_loss=d.train_on_batch(X_d_gan,y_d_gan)
            # train generator
            g_loss=dcgan.train_on_batch(X_g,y_g)
            show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])
        image = combine_images(g.predict(z_pred))
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
            .save(Generator_file+"%03depoch.png" % (epoch))
        # save models
        g.save('dcgan_generator.h5')
        d.save('dcgan_discriminator.h5')
        print()
if __name__ == '__main__':
    train()