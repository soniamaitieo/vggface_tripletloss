#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:18:11 2021

@author: sonia
"""

import tensorflow as tf
import os
import numpy as np


img_height, img_width = 224,224
BATCH_SIZE = 64

model_path = "/home/sonia/StatTypicalityHuman/results/2021-02-04-10-40/my_model"
new_model = tf.keras.models.load_model(model_path)

def embedding_model(model_path):
    new_model = tf.keras.models.load_model(model_path)
    new_model.summary()
    #new_model.pop()
    new_model = tf.keras.models.Sequential(new_model.layers[:-1])
    print("Architecture custom")
    print(new_model.summary())
    return(new_model)

    
mymodel = embedding_model(model_path)    
    
    
    
data_dir = "/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3Images"
#data_dir = "/media/sonia/DATA/CASIA90_TRAIN"


list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=False)


# get the count of image files in the test directory
image_count=0
for dir1 in os.listdir(data_dir):
    for files in os.listdir(os.path.join(data_dir, dir1)):
        image_count+=1
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)


#To process the label
def get_label(file_path):
  # convert the path to a list of path components separated by sep
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
# Integer encode the label
  return tf.argmax(tf.cast(one_hot, tf.int32))


# To process the image
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])


# To create the single testing of validation example with image and its corresponding label
def process_path(file_path):
  label = get_label(file_path)
# load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))

test_ds = list_ds



#test_ds = list_ds.take(int(image_count))

AUTOTUNE = tf.data.experimental.AUTOTUNE
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

test_ds = configure_for_performance(test_ds)

res_emb = mymodel.predict(test_ds)
np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-04-10-40" + "/CFDvecs.tsv", res_emb, delimiter='\t')

res_emb2 = new_model.predict(test_ds)
np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-04-10-40" + "/CFDvecs2.tsv", res_emb2, delimiter='\t')


MU = np.mean(res_emb2, axis=0)
SIGMA = np.cov(res_emb2, rowvar=0)
                           
from scipy.stats import multivariate_normal
var = multivariate_normal(MU, SIGMA , allow_singular=True )

pdftest=var.pdf(res_emb2)                           
log_pdftest= -np.log(pdftest)

np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-04-10-40" + "/log_pdftest2.tsv", log_pdftest, delimiter='\t')


def list_of_pict(dirName):
    """Get the list of all files in directory tree at given path"""
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        #listOfFiles.append([os.path.join(file) for file in filenames])
        for file in filenames:
            if file.endswith('.jpg'):
                listOfFiles.append(dirpath + '/' + file)
    return(listOfFiles)


#IMAGES - GET FULLPATH OF ALL IMAGES

full_path = list_of_pict(data_dir)
#Get name of indiv




y = multivariate_normal.pdf(res_emb2, mean=MU, cov=SIGMA)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

is_pos_def(SIGMA)