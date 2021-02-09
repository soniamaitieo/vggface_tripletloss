#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:26:51 2021

@author: sonia
"""

#--- LIBRARIES ---#
import tensorflow as tf
import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import sys
import matplotlib.pyplot as plt
from keras.callbacks import Callback
start = time.time()

#--- GLOBAL VAR ---#
img_height, img_width = 224,224
BATCH_SIZE = 64
inputShape = (224, 224)


data_dir = "/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3Images"
res_dir = "/home/sonia/StatTypicalityHuman/results/2021-02-05-16-13/"


#--- MODEL ---#

def embedding_model(model_path):
    new_model = tf.keras.models.load_model(model_path)
    new_model.summary()
    new_model.pop()
    new_model.pop()
    #new_model = tf.keras.models.Sequential(new_model.layers[:-1])
    print("Architecture custom")
    print(new_model.summary())
    return(new_model)

    
mymodel = embedding_model(res_dir)    
    
#--- DATA PROCESSING ---#
def list_of_pict(dirName):
    """Get the list of all files in directory tree at given path"""
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        #listOfFiles.append([os.path.join(file) for file in filenames])
        for file in filenames:
            if file.endswith('.jpg'):
                listOfFiles.append(dirpath + '/' + file)
    return(listOfFiles)

def all_preprocess_image(image_path):
    from keras_vggface import utils
    img =  tf.keras.preprocessing.image.load_img(image_path, target_size=inputShape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img =  utils.preprocess_input(img, version=1)
    return img

#Images (numpy) - dim: (nb, 224, 224, 3)
listpict = list_of_pict(dirName=data_dir)
img_data = list(map(all_preprocess_image, listpict))
img_data = np.array(img_data)
img_data=np.rollaxis(img_data,1,0)
img_data=img_data[0]

test_ds = img_data

print(tf.data.experimental.cardinality(test_ds).numpy())


#--- DATA INCREASE PERF---#

AUTOTUNE = tf.data.experimental.AUTOTUNE

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(BATCH_SIZE,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

test_ds = configure_for_performance(test_ds)


#--- PREDICT - FEATURES EXTRACTION---#
res_emb = mymodel.predict(test_ds)


#--- FIT MULTI GAUSS--#
MU = np.mean(res_emb, axis=0)
SIGMA = np.cov(res_emb, rowvar=0)
                           
from scipy.stats import multivariate_normal
var = multivariate_normal(MU, SIGMA , allow_singular=True )

pdftest=var.pdf(res_emb)                           
log_pdftest= -np.log(pdftest)

#np.savetxt( res_dir + "/log_pdftest.tsv", log_pdftest, delimiter='\t')

import pandas as pd


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

