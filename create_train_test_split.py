#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:58:43 2016

@author: matt
"""

import os, cv2
from sklearn.cross_validation import train_test_split
import tensorflow as tf
tf.python.control_flow_ops = tf

TRAIN_DIR = '/home/matt/kaggle/c_vs_d/input/train/'
TEST_DIR = '/home/matt/kaggle/c_vs_d/input/test/'

ROWS = 256
COLS = 256
CHANNELS = 1

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def save_test_image(images):
    count = len(images)
    for i, file_path in enumerate(images):
        img = read_image(file_path)
        cv2.imwrite(file_path.split('input')[0] + 'test' + str(ROWS) + 'g/' +
                    file_path.split('test/')[1], img)
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

def save_train_image(images, validation):
    count = len(images)
    for i, file_path in enumerate(images):
        img = read_image(file_path)
        if not validation:   
            if 'cat' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'train256g/cat/' + 
                            file_path.split('train/')[1], img)
            if 'dog' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'train256g/dog/' + 
                            file_path.split('train/')[1], img)
        else:
            if 'cat' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'validation256g/cat/' + 
                            file_path.split('train/')[1], img)
            if 'dog' in file_path:
                cv2.imwrite(file_path.split('input')[0] + 'validation256g/dog/' + 
                            file_path.split('train/')[1], img)
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

# Maintain aspect ratios and reflect borders
def read_image(file_path): # max image width = 1050, max height = 768
    img = cv2.imread(file_path, 0)
    shape = img.shape
    h, w = shape[0], shape[1]
    asp_ratio = w*1./h
    if h>=ROWS or w>=COLS:
        if h>w:
            img = cv2.resize(img, (int(COLS*asp_ratio), ROWS), cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, (COLS, int(ROWS*1./asp_ratio)), 
                             cv2.INTER_CUBIC)
    shape = img.shape
    h, w = shape[0], shape[1]
    h_diff = ROWS-h
    w_diff = COLS-w
    h_rem = h_diff%2
    w_rem = w_diff%2
    img = cv2.copyMakeBorder(img, h_diff/2+h_rem, h_diff/2, 
                              w_diff/2+w_rem, w_diff/2, 
                              cv2.BORDER_REFLECT)
    return img

# Maintain proportion of cats to dogs in training and validation sets
X_train, X_test, y_train, y_test = train_test_split(train_dogs, len(train_dogs) * [1], 
                                                    test_size=0.1, 
                                                    random_state=99)
save_train_image(X_train, False)
save_train_image(X_test, True)

X_train, X_test, y_train, y_test = train_test_split(train_cats, len(train_cats) * [1], 
                                                    test_size=0.1, 
                                                    random_state=99)
save_train_image(X_train, False)
save_train_image(X_test, True)
save_test_image(test_images)