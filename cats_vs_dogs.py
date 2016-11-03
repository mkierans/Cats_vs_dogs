#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:58:43 2016

@author: matt
"""

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D
from keras.layers import MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop, Nadam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import tensorflow as tf
tf.python.control_flow_ops = tf

train_dir = '/home/matt/kaggle/c_vs_d/train/'
validation_dir = '/home/matt/kaggle/c_vs_d/validation/'

ROWS = 256
COLS = 256
CHANNELS = 3

def conv_model():
    nb_train_samples = 20000
    nb_validation_samples = 5000
    nb_filters = 32
    nb_epoch = 1000
    batch_size = 16
#    print type(X_train), type(y_train)
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_row=5, nb_col=5, 
                            border_mode='same', dim_ordering='th', 
                            input_shape=(CHANNELS, ROWS, COLS), 
                            activation='relu'))
    model.add(Dropout(0.05))
    model.add(Convolution2D(nb_filters, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu',
                            dim_ordering='th')) #256x256
    model.add(Dropout(0.05))
    model.add(Convolution2D(nb_filters*2, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th', subsample=(2, 2))) #128x128
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*4, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th', subsample=(2, 2))) #64x64
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th', subsample=(2, 2))) #32x32
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th', subsample=(2, 2))) #16x16
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu', 
                            dim_ordering='th', subsample=(2, 2))) #8x8
    model.add(Dropout(0.2))
#    model.add(Convolution2D(nb_filters*5, nb_row=3, nb_col=3, 
#                            border_mode='same', activation='relu',
#                            subsample = (2, 2))) #16x16
#    model.add(Dropout(0.2))
#    model.add(Convolution2D(nb_filters*4, nb_row=3, nb_col=3, 
#                            border_mode='same', activation='relu',
#                            subsample = (2, 2))) #8x8
#    model.add(Dropout(0.3))
#    model.add(Convolution2D(nb_filters*4, nb_row=3, nb_col=3, 
#                            border_mode='same', activation='relu',
#                            subsample = (2, 2))) #4x4
#    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), 
                  metrics=['accuracy'])
    train_dataGen = ImageDataGenerator(featurewise_center=False, #Set input mean to 0 over the dataset
               samplewise_center=False, #Set each sample mean to 0
               featurewise_std_normalization=False, # Divide inputs by std of the dataset.
               samplewise_std_normalization=False, # Divide each input by its std.
#               zca_whitening=True, # Apply ZCA whitening.
               rotation_range=45, # Degree range for random rotations
#               width_shift_range=0, # Float (fraction of total width) Range for random horizontal shifts
#               height_shift_range=0, # Float (fraction of total height). Range for random vertical shifts
#               shear_range=0.0, # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
               zoom_range=0.2, # Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
#               channel_shift_range=0, # Float. Range for random channel shifts.
#               fill_mode='reflect', # One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
#               cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
               horizontal_flip=True, # Randomly flip inputs horizontally.
               vertical_flip=True, # Randomly flip inputs vertically.
               rescale = 1./255, # Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
               dim_ordering='th') # One of {"th", "tf"}. "tf" mode means that the images should have shape )
#    train_dataGen.fit(X_sample) # Must run this if using featurewise_center, featurewise_std_normaization, or zca_whitening
    val_dataGen = ImageDataGenerator(rescale=1./255, dim_ordering='th')
    train_generator = train_dataGen.flow_from_directory(train_dir,
                                                        target_size=(ROWS, COLS), 
                                                        batch_size=batch_size, 
                                                        class_mode='binary')
    val_generator = val_dataGen.flow_from_directory(validation_dir,
                                                    target_size=(ROWS, COLS), 
                                                    batch_size=batch_size,
                                                    class_mode='binary')
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 100, verbose = 1, mode = 'auto')
    model.fit_generator(train_generator, nb_epoch=nb_epoch, samples_per_epoch=nb_train_samples,
                                     validation_data = val_generator,
                                     callbacks = [checkpointer, earlystopping],
                                     nb_val_samples=nb_validation_samples)
#    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32) 
#    print loss_and_metrics
    return model

model = conv_model()
    






#def show_cats_and_dogs(idx):
#    cat = read_image(train_cats[idx])
#    dog = read_image(train_dogs[idx])
#    pair = np.concatenate((cat, dog), axis=1)
#    plt.figure(figsize=(10,5))
#    plt.imshow(pair)
#    plt.show()
#    
#for idx in range(0,5):
#    show_cats_and_dogs(idx)
    



