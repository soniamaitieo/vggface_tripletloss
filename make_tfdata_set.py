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
"""
def class_names(data_dir):
    class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))
    return(class_names)
"""

#Convert path to (img, label) tuple
def get_label(file_path, class_names):
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



def process_path(file_path,class_names):
  label = get_label(file_path,class_names)
  img = tf.io.read_file(file_path)  # load the raw data from the file as a string
  img = decode_img(img)
  return img, label


def process_gender(filename, label):
    #filename = tf.read_file(filename)
    img = tf.io.read_file(filename)  # load the raw data from the file as a string
    img = decode_img(img)
    return img, label



def choose_ds(list_ds , choice, image_count,n_train,n_val,n_test):
    if choice =="close":
        #CLOSE-DS#
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        train_ds = list_ds.take(n_train)
        val_ds = list_ds.skip(n_train).take(n_val)
        test_ds = list_ds.skip(n_train + n_val).take(n_test)
        return(train_ds,val_ds,test_ds)
    if choice =="semiclose":
        #SEMI-CLOSE-DS#
        test_ds = list_ds.take(n_test)
        close_ds = list_ds.skip(n_test).take(n_train + n_val)
        close_ds = close_ds.shuffle(n_train + n_val , reshuffle_each_iteration=False)
        train_ds = close_ds.take(n_train)
        val_ds = close_ds.skip(n_train).take(n_val)
        return(train_ds,val_ds,test_ds)
    if choice =="open":
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
    rd_angle = round(rd.uniform(0, 0.5), 2)
    img = tfa.image.transform_ops.rotate(image, rd_angle)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, 0.3)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    return (img, label)


AUTOTUNE = tf.data.experimental.AUTOTUNE


# optimze preprocessing performance
def configure_for_performance(ds):
  ds = ds.cache()
  #ds = ds.shuffle(buffer_size=n_train)
  ds = ds.map(augment, num_parallel_calls=AUTOTUNE) # augmentation call
  ds = ds.batch(batch_size=128,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


################################################################################
##### make new file VGGmodel.py
from keras_vggface.vggface import VGGFace

def model_vgg16_verif(dr):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)



def model_vgg16_classif(dr,class_names):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

################################################################################
#### topNacc.py & mapNacc.py

import sklearn.metrics.pairwise
from keras.callbacks import Callback
from keras import Model


def calc_TOPacc(distance_array, val_ds, TOP = 5):
    y = np.concatenate([y for x, y in val_ds], axis=0)
    #print(y)
    TOPacc = []
    #labels = valid_generator.filenames
    for i in range(len(distance_array)):
        L=[]
        #i = 1
        distance_i = distance_array[i,:]
        #idx = np.argpartition(distance_i, TOP )
        L=  list(np.argpartition(distance_i,range(TOP+1))[1:TOP+1])
        #This returns the k-smallest values. Note that these may not be in sorted order.
        #if i in L: L.remove(i)
        #print("current_indiv:" , y[i])
        #print("Top_3_img" , L)
        #L = list(distance_i[idx[:TOP]])
        #print(itemgetter(*L)(labels))
        #print("Top_3_indiv" , [y[j] for j in L])
        if y[i] in [y[j] for j in L]:
            TOPacc.append(1)
        else:
            TOPacc.append(0)
    return(sum(TOPacc)/len(TOPacc))


all_TOP3_acc = []
## test compared to train images
class TopX(Callback):
    def __init__(self, x):
          self.x = x
          #self.topX = []
    def on_train_begin(self, logs={}):
          self.topX = []
    def on_epoch_end(self, epoch, logs={}):
        TOP = 5
        #distance_array = compute_distance(self.model,self.x)
        distance_array = sklearn.metrics.pairwise_distances(self.model.predict(self.x,verbose=1), metric='l2')
        # = sklearn.metrics.pairwise.cosine_similarity(self.x,self.x)
        #print(self.x.filenames)
        topX = calc_TOPacc(distance_array, self.x, TOP = TOP)
        #print("Top {TOP} - Accuracy(%):   ", round(topX*100,1))
        print(' Top {} - Accuracy {} %'.format(TOP, round(topX*100,1)))
        all_TOP3_acc.append(round(topX*100,1))
        #print(topX)




# We make a dictionnary where the index is the number of the individual, and the value is a list with the position of associated images into a list
def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


#This function sample the image-anchor for the query indiv and 1 image for each indiv
def sample_candidates_dist(curr_ind , y_dict):
    pict_for_each_ind = dict((k, rd.choice(v)) for k, v in y_dict.items())
    Q = pict_for_each_ind[curr_ind] #GET IMG POS OF CURR_IND (this is the "posititive image")
    all_A = y_dict[curr_ind][:] #to avoid mutation in original dict add [:]
    all_A.remove(Q) #take another diff image Q != A
    A = rd.choice(all_A)
    A = (curr_ind,A)
    #return dict of candidates (key=ind,value=pos_img) and the tupe anchor (ind,pos_img)
    return((pict_for_each_ind,A))


def calc_ap_per_ind(res_val,curr_ind, y_dict,TOP=5):
    #get candidates immg
    all_img, anchor = sample_candidates_dist(curr_ind, y_dict = y_dict)
    #Get embedding of candidates images
    res_anchor = res_val[anchor[1], :].reshape(1, -1) #reshape to have compatible dim to compare
    res_comp = res_val[list(all_img.values()),:]
    candidates_comp = list(all_img.keys())
    distance_array_with_A = (sklearn.metrics.pairwise_distances(res_comp, res_anchor, metric='l2')).ravel()
    #Attention L-top va donner le rang des elements qui ne match pas avec le numéro de l'individu, d'où la necessité de garder les num individus dans candidates_comp
    L_top= np.argpartition(distance_array_with_A ,range(TOP))[:TOP]
    tp_counter = 0
    cumulate_precision = 0
    av_precision = 0
    for i in range(len(L_top)):
        #check si l'autre img de lindiv est dans le TOP
        if anchor[0] == candidates_comp[L_top[i]]:
            #compte le nombre de match
            tp_counter += 1
            cumulate_precision += (float(tp_counter)/float(i+1))
    if tp_counter != 0:
        av_precision = cumulate_precision/1 #here we have only 1 positive img
    else:
        av_precision = 0
    return(av_precision)



mAP_history=[]
class map_N(Callback):
    def __init__(self, x):
        self.x = x
    def on_train_begin(self, logs={}):
        self.all100_mAP = []
        self.TOP = 1
    def on_epoch_end(self, epoch, logs={}):
        res_val = self.model.predict(self.x,verbose=1) #embedding
        y_label = list(np.concatenate([y for x, y in self.x], axis=0))
        y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
        y_label
        for it in range(100) :
            mAP_each_ind= [calc_ap_per_ind(res_val,i, y_dict,TOP=self.TOP) for i in list(y_dict.keys())]
            self.all100_mAP.append(np.mean(mAP_each_ind))
        print(' MAP@{} - Accuracy {} %'.format(self.TOP, round(np.mean(self.all100_mAP)*100,1)))
        mAP_history.append(round(np.mean(self.all100_mAP)*100,1))


class map_N_classif(Callback):
    def __init__(self, x):
        self.x = x
    def on_train_begin(self, logs={}):
        self.all100_mAP = []
        self.TOP = 5
    def on_epoch_end(self, epoch, logs={}):
        feature_network = Model(self.model.input, self.model.get_layer('feature').output)
        res_val = feature_network.predict(self.x)
        #res_val = self.model.predict(self.x,verbose=1) #embedding
        y_label = list(np.concatenate([y for x, y in self.x], axis=0))
        y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
        y_label
        for it in range(100) :
            mAP_each_ind= [calc_ap_per_ind(res_val,i, y_dict,TOP=self.TOP) for i in list(y_dict.keys())]
            self.all100_mAP.append(np.mean(mAP_each_ind))
        print(' MAP@{} - Accuracy {} %'.format(self.TOP, round(np.mean(self.all100_mAP)*100,1)))
        mAP_history.append(round(np.mean(self.all100_mAP)*100,1))
        
        
        
##############################################################################

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.001
  if epoch > 20:
    learning_rate = 0.0001
  if epoch > 70:
    learning_rate = 0.00001
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

