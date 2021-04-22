#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:11:53 2021

@author: sonia
"""


import os
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import tensorflow_addons as tfa


#--- DATA PROCESSING ---#

def class_names(data_dir):
    class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))
    return(class_names)
"""
#Convert path to (img, label) tuple
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)  # convert the path to a list of path components
  one_hot = parts[-2] == class_names  # The second to last is the class-directory
  return tf.argmax(tf.cast(one_hot, tf.int32))

# To process the image
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  resized_image = tf.image.resize(img, [224, 224])
  final_image = tf.keras.applications.vgg16.preprocess_input(resized_image)
  #final_image = tf.keras.applications.resnet.preprocess_input(resized_image)
  #final_image = tf.keras.applications.resnet50.preprocess_input(resized_image)
  return final_image


def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)  # load the raw data from the file as a string
  img = decode_img(img)
  return img, label
"""

def choose_ds(list_ds , choice, image_count,n_train,n_val,n_test):
    if choice =="close":
        #CLOSE-DS#
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        train_ds = list_ds.take(n_train)
        val_ds = list_ds.skip(n_train).take(n_val)
        test_ds = list_ds.skip(n_train + n_val).take(n_test)
        return(train_ds,val_ds,test_ds)
    if choice =="close":
        #SEMI-CLOSE-DS#
        test_ds = list_ds.take(n_test)
        close_ds = list_ds.skip(n_test).take(n_train + n_val)
        close_ds = close_ds.shuffle(n_train + n_val , reshuffle_each_iteration=False)
        train_ds = close_ds.take(n_train)
        val_ds = close_ds.skip(n_train).take(n_val)
        return(train_ds,val_ds,test_ds)
    if choice =="close":
        #OPEN-DS#
        train_ds = list_ds.take(n_train)
        val_ds = list_ds.skip(n_train).take(n_val)
        test_ds = list_ds.skip(n_train + n_val).take(n_test)
        return(train_ds,val_ds,test_ds)
    

    

def dict_set_ind_img(train_ds,val_ds,test_ds):
    """
    Return 1 dict of 3 dict (train, val,test) with number of images for each ds
    """
    label_tmp = [y.numpy() for x, y in train_ds]
    train_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) ) 
    label_tmp = [y.numpy() for x, y in val_ds]
    val_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) ) 
    label_tmp = [y.numpy() for x, y in test_ds]
    test_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) ) 
    d = {
        "train": train_dict,
        "val": val_dict,
        "test" : test_dict,
    }
    return(d)

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def dict_ind_img(train_ds,val_ds,test_ds):
    """
    Return 1 dict of 3 dict (train, val,test) with list of all images for each indiv for each ds
    """
    label_tmp = [y.numpy() for x, y in train_ds]
    train_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) ) 
    label_tmp = [y.numpy() for x, y in val_ds]
    val_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) ) 
    label_tmp = [y.numpy() for x, y in test_ds]
    test_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) ) 
    d = {
        "train": train_dict,
        "val": val_dict,
        "test" : test_dict,
    }
    return(d) 


def augment(image, label):
    rd_angle = round(rd.uniform(0, 0.2), 2)
    img = tfa.image.transform_ops.rotate(image, rd_angle)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, 0.3)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    return (img, label)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# optimze preprocessing performance
def configure_for_performance(ds,batch_size):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.map(augment, num_parallel_calls=AUTOTUNE) # augmentation call
  ds = ds.batch(batch_size,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds



