#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:58:43 2016

@author: matt
"""

import matplotlib.pyplot as plt
import os, cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D
from keras.layers import MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop, Nadam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf

K.set_image_dim_ordering('th')
train_dir = '/home/matt/kaggle/c_vs_d/train256g/'
validation_dir = '/home/matt/kaggle/c_vs_d/validation256g/'
#cat_dir = '/home/matt/kaggle/c_vs_d/train/cat/'
#dog_dir = '/home/matt/kaggle/c_vs_d/train/dog/'
#img_samp = [cat_dir+i for i in os.listdir(cat_dir)[0:1000]]
#img_samp = img_samp + [dog_dir+i for i in os.listdir(dog_dir)[0:1000]]

#def read_image(file_path):
#    return cv2.imread(file_path, 1)
#    
#def prep_data(images):
#    count = len(images)
#    data = np.ndarray((count, CHANNELS, ROWS, COLS), # Figure this out!!!!!)
#    for i, image_file in enumerate(images):
#        image = read_image(image_file)
#        data[i] = image.T
#        if i%250 == 0: print('Processed {} of {}'.format(i, count))
#    return data
#
#X_samp = prep_data(img_samp)

ROWS = 256
COLS = 256
CHANNELS = 1

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def conv_model():
    nb_train_samples = 20000
    nb_validation_samples = 5000
    nb_filters = 16
    nb_epoch = 1000
    batch_size = 16
#    print type(X_train), type(y_train)
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_row=3, nb_col=3, #256x256
                            border_mode='same', 
                            input_shape=(CHANNELS, ROWS, COLS), 
                            activation='relu'))
#    model.add(Dropout(0.05))
    model.add(Convolution2D(nb_filters, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
#    model.add(Dropout(0.05))    
    model.add(MaxPooling2D(pool_size=(2, 2))) #128x128
    model.add(Convolution2D(nb_filters*2, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu')) 
#    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*2, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
#    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2))) #64x64
    model.add(Convolution2D(nb_filters*4, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filters*4, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2))) #32x32
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*8, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2))) #16x16
    model.add(Convolution2D(nb_filters*16, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*16, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))   
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2))) #8x8
    model.add(Convolution2D(nb_filters*32, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*32, nb_row=3, nb_col=3, 
                            border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #4x4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
#    model.load_weights('/home/matt/kaggle/c_vs_d/weights256g97.hdf5')
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=1e-4), 
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
#               zoom_range=0.2, # Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
#               channel_shift_range=0, # Float. Range for random channel shifts.
#               fill_mode='reflect', # One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
#               cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
               horizontal_flip=True, # Randomly flip inputs horizontally.
               vertical_flip=True, # Randomly flip inputs vertically.
               rescale = 1./255) # Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
#               dim_ordering='th') # One of {"th", "tf"}. "tf" mode means that the images should have shape )
#    train_dataGen.fit(X_samp) # Must run this if using featurewise_center, featurewise_std_normaization, or zca_whitening
    val_dataGen = ImageDataGenerator(rescale=1./255)
    train_generator = train_dataGen.flow_from_directory(train_dir,
                                                        target_size=(ROWS, COLS), 
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        class_mode='categorical')
    val_generator = val_dataGen.flow_from_directory(validation_dir,
                                                    target_size=(ROWS, COLS), 
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')
    history = LossHistory()
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 25, verbose = 1, mode = 'auto')
    model.fit_generator(train_generator, nb_epoch=nb_epoch, samples_per_epoch=nb_train_samples,
                                     validation_data = val_generator,
                                     callbacks = [checkpointer, earlystopping, history],
                                     nb_val_samples=nb_validation_samples)
    return model, history

model, history = conv_model()

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,len(loss))[0::2])
plt.legend()
plt.show()