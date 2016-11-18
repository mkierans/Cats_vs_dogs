#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:09:13 2016

@author: matt
"""

import pandas as pd
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf
K.set_image_dim_ordering('tf')
         
model = load_model('/home/matt/kaggle/c_vs_d/model_VGG16.h5')
model.load_weights('/home/matt/kaggle/c_vs_d/weights_VGG16.hdf5')

IMSIZE = (224, 224)
CHANNELS = 3

target_size = (224, 224, CHANNELS)

test_dir = '/home/matt/kaggle/c_vs_d/input/test/'
test_images =  [test_dir+i for i in os.listdir(test_dir+'test/')]

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=IMSIZE,
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle=False)

images = test_images
pred = model.predict_generator(test_generator, len(images))

ids = np.array([images]).T
label = np.array(pred)

ids_df = pd.DataFrame(ids, columns=['id'])

''' Use this is there is one predicted class'''
#label_df = pd.DataFrame(label, columns=['label'])
#df = pd.concat([ids_df, label_df], axis=1)

''' Use this if there are two predicted classes '''
label_df = pd.DataFrame(label, columns=['cat', 'label'])
label_df = label_df.div(label_df.sum(axis=1), axis=0)
df = pd.concat([ids_df, label_df.drop('cat', axis=1)], axis=1)

df.id = df.id.apply(lambda x: int(x.split('test/')[1].split('.jpg')[0]))
df.to_csv('VGG16.csv', index=False)
