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

model_path = "/home/sonia/StatTypicalityHuman/results/2021-02-05-16-13/my_model"
new_model = tf.keras.models.load_model(model_path)

def embedding_model(model_path):
    new_model = tf.keras.models.load_model(model_path)
    new_model.summary()
    new_model.pop()
    new_model.pop()
    #new_model = tf.keras.models.Sequential(new_model.layers[:-1])
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
  file_name = parts[-1]
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


#test_ds = list_ds.take(int(image_count))

test_ds = list_ds
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


y = np.concatenate([y for x, y in test_ds], axis=0)

#truc = class_names[y]
res_emb = mymodel.predict(test_ds)

#np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-04-10-40" + "/CFDvecs.tsv", res_emb, delimiter='\t')


MU = np.mean(res_emb, axis=0)
SIGMA = np.cov(res_emb, rowvar=0)
                           
from scipy.stats import multivariate_normal
var = multivariate_normal(MU, SIGMA , allow_singular=True )

pdftest=var.pdf(res_emb)                           
log_pdftest= -np.log(pdftest)

#np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-05-16-13/" + "/log_pdftest.tsv", log_pdftest, delimiter='\t')

import pandas as pd
#from pandas import ExcelWriter
#from pandas import ExcelFile

#READ TAB
df = pd.read_excel('/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3NormingDataandCodebook.xlsx',
                   sheet_name='CFD2.0.3NormingData', engine='openpyxl')


df = df.drop([0,1,2])
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(3))

#SELECT
df = df[['Target' , 'Gender', 'Feminine' , 'Attractive' ]]


stat_typ =  pd.DataFrame()
stat_typ["Target"] = class_names[y]
stat_typ["LL"] = log_pdftest


stat_typ = stat_typ.groupby(['Target']).mean()

all_df = pd.merge(df, stat_typ , on='Target')


import matplotlib.pyplot as plt
plt.scatter(all_df['Attractive'], all_df['LL'])
from scipy import stats
corr = stats.pearsonr(all_df['Attractive'], all_df['LL'])
r = np.corrcoef(all_df['Attractive'], all_df['LL'])


"""
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

is_pos_def(SIGMA)
"""


import tensorflow as tf

def all_preprocess_image(image_path):
    from keras_vggface import utils
    img =  tf.keras.preprocessing.image.load_img(image_path, target_size=inputShape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img =  utils.preprocess_input(img)
    return img


TEST= map(all_preprocess_image, full_path)
test = list(TEST)

test_tf = tf.convert_to_tensor(test, np.float32)

res_emb2 = mymodel.predict(test)
