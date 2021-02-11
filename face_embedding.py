#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:18:11 2021

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
#res_dir = "/home/sonia/StatTypicalityHuman/results/2021-02-11-10-20"
#res_dir = "/home/sonia/StatTypicalityHuman/results/2021-02-11-15-4/"
res_dir = "/home/sonia/StatTypicalityHuman/results/2021-02-11-15-28"

#--- MODEL ---#

def embedding_model(model_path):
    new_model = tf.keras.models.load_model(model_path)
    new_model.summary()
    new_model.pop()
    #new_model = tf.keras.models.Sequential(new_model.layers[:-1])
    print("Architecture custom")
    print(new_model.summary())
    return(new_model)

    
mymodel = embedding_model(res_dir + "/my_model")  
  
mymodel = tf.keras.models.load_model(res_dir + "/my_model")
    
#--- DATA PROCESSING ---#

list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*-N.jpg'), shuffle=False)

# get the count of image files in the test directory
image_count= len([i for i in list_ds ])


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
  resized_image = tf.image.resize(img, [224, 224])
  final_image = tf.keras.applications.vgg16.preprocess_input(resized_image)
  return final_image

# To create the single testing of validation example with image and its corresponding label
def process_path(file_path):
  label = get_label(file_path)
# load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))


test_ds = list_ds


#--- DATA INCREASE PERF---#

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


#--- PREDICT - FEATURES EXTRACTION---#
#truc = class_names[y]
res_emb = mymodel.predict(test_ds)
label = class_names[y]
np.savetxt(res_dir + "/CFDvecs.tsv", res_emb, delimiter='\t')



#--- READ CFD AND EXTRACT INFORMATIVE COL ---#
import pandas as pd
df = pd.read_excel('/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3NormingDataandCodebook.xlsx',
                   sheet_name='CFD2.0.3NormingData', engine='openpyxl')
df = df.drop([0,1,2])
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(3))
#SELECT
df = df[['Target' , 'Gender', 'Feminine' , 'Attractive' ]]
df = df.dropna()

L = []
for i in df.Target :
    print(i)
    pos_i = np.where(label == i)[0][0]
    emb_i = res_emb[[pos_i], :]
    emb_without_i = np.delete(res_emb, (pos_i), axis=0)
    MU = np.mean(emb_without_i, axis=0)
    SIGMA = np.cov(emb_without_i, rowvar=0)                      
    from scipy.stats import multivariate_normal
    var = multivariate_normal(MU, SIGMA , allow_singular=True )
    #var = multivariate_normal(MU, SIGMA  )
    pdftest=var.pdf(emb_i)                           
    log_pdftest= np.log(pdftest)
    print(log_pdftest)
    L.append(log_pdftest)




df["LL"] = L

P = []
from scipy import stats
for dim in range (90):
    plt.hist(res_emb[:, dim]) 
    k2, p = stats.normaltest(res_emb[:, dim])
    P.append(p)
toto = [o>0.005 for o in P]
    
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

#np.savetxt("/home/sonia/StatTypicalityHuman/results/2021-02-10-14-21/" + "/log_pdftest.tsv", stat_typ["Target"], delimiter='\t')
stat_typ.to_csv("/home/sonia/StatTypicalityHuman/results/2021-02-10-14-21/" + "/log_pdftest.csv")

stat_typ = stat_typ.groupby(['Target']).mean()

all_df = df



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
