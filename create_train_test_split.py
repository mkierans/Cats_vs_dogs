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

from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D
from keras.layers import MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import tensorflow as tf
tf.python.control_flow_ops = tf

TRAIN_DIR = '/home/matt/kaggle/c_vs_d/input/train/'
TEST_DIR = '/home/matt/kaggle/c_vs_d/input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def save_image(images, validation):
    count = len(images)
    for i, file_path in enumerate(images):
        img = read_image(file_path)
        if not validation:   
            if 'cat' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'train64/cat/' + 
                            file_path.split('train/')[1], img)
            if 'dog' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'train64/dog/' + 
                            file_path.split('train/')[1], img)
        else:
            if 'cat' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'validation64/cat/' + 
                            file_path.split('train/')[1], img)
            if 'dog' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'validation64/dog/' + 
                            file_path.split('train/')[1], img)
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
                
def read_image(file_path): # max image width = 1050, max height = 768
    img = cv2.imread(file_path, 1)
    shape = img.shape
    h, w = shape[0], shape[1]
    if h>=ROWS and w>=COLS:
        return cv2.resize(img, (ROWS, COLS), cv2.INTER_CUBIC)
    elif h>=ROWS and w<COLS:
        resize = cv2.resize(img, (w, ROWS), cv2.INTER_CUBIC)
        w_diff = COLS-w
        w_rem = w_diff%2
        return cv2.copyMakeBorder(resize, 0, 0, w_diff/2+w_rem, w_diff/2,
                                  cv2.BORDER_REFLECT)
    elif h<ROWS and w>=COLS:
        resize = cv2.resize(img, (COLS, h), cv2.INTER_CUBIC)
        h_diff = ROWS-h
        h_rem = h_diff%2
        return cv2.copyMakeBorder(resize, h_diff/2+h_rem, h_diff/2, 0, 0,
                                  cv2.BORDER_REFLECT)
    h_diff = ROWS-h
    w_diff = COLS-w
    h_rem = h_diff%2
    w_rem = w_diff%2
#    print h_diff, w_diff, file_path
    return cv2.copyMakeBorder(img, h_diff/2+h_rem, h_diff/2, 
                              w_diff/2+w_rem, w_diff/2, 
                              cv2.BORDER_REFLECT)

def prep_data(images):
    count = len(images)
#    data = np.ndarray((count, CHANNELS, COLS, ROWS), dtype=np.uint8)
    for image_file in enumerate(images):
        image = read_image(image_file)
#        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    return data

#train = prep_data(train_images)
#test = prep_data(test_images[0:100])

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1.0)
    else:
        labels.append(0.0)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(train_images, labels, 
                                                    test_size=0.2, 
                                                    random_state=99)
save_image(X_train, False)
save_image(X_test, True)

