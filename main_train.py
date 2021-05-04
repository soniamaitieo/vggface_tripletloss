#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:21:08 2021

@author: sonia
"""


#--- LIBRRAIRIES ---#
import sys
import io
import os
import pathlib
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda
from keras_vggface.vggface import VGGFace
import sklearn.metrics.pairwise
from keras.callbacks import Callback
import random as rd
import pandas as pd
from make_tfdata_set import *


#--- GLOBAL PARAMETERS ---#
img_height = 224
img_width = 224
batch_size = 128
n_epochs = 100
LR = 0.001
LR2 = 0.00001
NOFREEZE = 3
dr = 0.5



data_dir = "/home/sonia/CASIA_minitrain/CASIA_minitrain"
data_dir = pathlib.Path(data_dir)
#class_names(data_dir)
class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))


#--- DATA PROCESSING ---#
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'), shuffle=False)



AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


#list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

#list_ds = truc(class_names, list_ds)
#list_ds = list_ds.map(lambda x: process_path(x, class_names), num_parallel_calls=AUTOTUNE)


#--- on va faire une triplet loss avec 2 catégories H/F ---#
imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
labels = [i.split('/')[-2] for i in imgs]
filenames = tf.constant(imgs)
labels = tf.constant(labels)

list_ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

list_ds = list_ds.map(process_gender)


image_count = len(list(data_dir.glob('*/*.jpg')))


n = image_count
n_train = int(n * 0.7)
n_val = int(n * 0.1)
#n_test= int(n * 0.01)
n_test = n - n_train - n_val #0.2

train_ds,val_ds,test_ds=choose_ds(list_ds , "semiclose", image_count,n_train,n_val,n_test)  


train_ds = configure_for_performance(train_ds)
val_ds = val_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE) 
test_ds =test_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)

model = model_vgg16_classif(dr,class_names)

mAP_history=[]

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    #optimizer = tf.keras.optimizers.Adam(LR,beta_1=0.02,beta_2=0.02),
    #loss=tfa.losses.TripletSemiHardLoss())
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['sparse_categorical_accuracy'])

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=n_epochs
   #callbacks=[map_N(val_ds),map_N(train_ds)]
   #callbacks = [map_N_classif(val_ds),map_N_classif(train_ds)]
    )

