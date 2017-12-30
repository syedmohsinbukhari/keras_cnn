#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:10:55 2017

@author: elcid
"""

import keras
import pickle
import numpy as np
from os.path import join

## INITIALIZE
np.random.seed(0)
MAX_EPOCHS = 1000

## IMPORT DATA
TRAIN_FILES = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4']
VAL_FILES = ['data_batch_5']
TEST_FILES = ['test_batch']
DATA_FOLDER = '/path/to/data/cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def get_data(f_names):
    features = np.empty((0, 3072), dtype = np.float32)
    labels = []
    for file_name in f_names:
        file_path = join(DATA_FOLDER, file_name)
        file_content = unpickle(file_path)
        
        data_features = file_content[b'data']
        data_labels = file_content[b'labels']
        
        features = np.concatenate((features, data_features))
        labels = labels + data_labels
    
    labels_ = keras.utils.to_categorical(labels, num_classes=10)
    features = features/255.0
    return features, labels_

train_features, train_labels = get_data(TRAIN_FILES)
val_features, val_labels = get_data(VAL_FILES)
test_features, test_labels = get_data(TEST_FILES)

## DEFINE MODEL
model = keras.Sequential()
model.add(keras.layers.Reshape((3, 32, 32), input_shape=(3072,)))
model.add(keras.layers.Conv2D(32, 2, activation='relu', \
                               data_format='channels_first'))
model.add(keras.layers.MaxPool2D(data_format='channels_first'))
model.add(keras.layers.Conv2D(64, 2, activation='relu', \
                               data_format='channels_first'))
model.add(keras.layers.MaxPool2D(data_format='channels_first'))
model.add(keras.layers.Conv2D(128, 2, activation='relu', \
                               data_format='channels_first'))
model.add(keras.layers.MaxPool2D(data_format='channels_first'))
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(units=50, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))

## MODEL COMPILATION
optmzr = keras.optimizers.nadam()
model.compile(loss='categorical_crossentropy',
               optimizer=optmzr,
               metrics=['accuracy'])

## CALLBACKS
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', \
                                               min_delta=0.001, patience=3, \
                                               verbose=0, mode='auto')
check_pointer = keras.callbacks.ModelCheckpoint('weights_cnn.hdf5', \
                                                save_best_only = True, \
                                                save_weights_only = True)

## FIT MODEL
hist = model.fit(x=train_features, y=train_labels, epochs=MAX_EPOCHS, \
                 validation_data=(val_features, val_labels),
                 callbacks=[early_stopping, check_pointer])

## EVALUATE MODEL
model.load_weights('weights_cnn.hdf5')
loss_and_metrics_2 = model.evaluate(test_features, test_labels)
print('TESTING\nLoss: {}\nAccuracy: {}'.format(loss_and_metrics_2[0], \
      loss_and_metrics_2[1]))