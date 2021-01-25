#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:48:20 2021

@author: steio
"""
#--- LIBRARIES ---#
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import sys 
from operator import itemgetter 
from keras.callbacks import Callback
import scipy.spatial


import tensorflow_datasets as tfds
train_dataset, test_dataset = tfds.load(name="mnist")


#--- GLOBAL VAR ---#
inputShape = (224, 224)
img_height, img_width = 224,224
#num_classes = 1
NOFREEZE = 2
BATCH_SIZE = 128
NB_EPOCHS = 10
#NB_EPOCHS2 = 5
LR = 0.001
opt = tf.keras.optimizers.Adam(learning_rate= LR)

#--- SAVE PARAMETERS and INFOS ---#
today =  datetime.date.today()
now = datetime.datetime.now()


#---create_folder for results
todaystr = today.isoformat() + '-' + str(now.hour) + '-' + str(now.minute)
os.mkdir("results/"   +  todaystr )

#---create log file to save all steps and outputs
log_file_path = "results/" + todaystr + "/log_file.txt"
sys.stdout = open(log_file_path, 'w')


#---Write a memo file with parameters used for CNN

#OPTIM2 = "keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)"
params_file_path =  "results/" + todaystr + "/params.txt"
f = open(params_file_path,'w')
f.write("weights:" + "VGGface " + "\n")
f.write("epochs:" + str(NB_EPOCHS) + "\n")
#f.write("epochs2:" + str(NB_EPOCHS2) + "\n")
f.write("batch_size:" + str(BATCH_SIZE) + "\n" )
f.write("optim1:" + str(type(opt))  + "\n")
f.write("optim1_LR:" + str(LR) + "\n")
#f.write("optim2:" + OPTIM2  + "\n")
f.write("no freezen layer:" + str(NOFREEZE) + "\n")
#f.write("Img_preprocessing" + "rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)")
f.write("Img_preprocessing:" + "rescale=1./255, horizontal_flip=False, rotation_range=20,width_shift_range=0.05, height_shift_range=0.05,brightness_range=[0.8,1.2], shear_range=0.1,zoom_range=[0.85,1.15]")
#f.write("val:" + str(0.20) + "\n")
#f.write("early_stop:" + str(10) + "\n")
#f.write("reducelr0.1:" + str(5) + "\n")
#f.write("dropout:" + str(0.5) + "\n")
#f.write("HorizontalFlip:" + "True" + "\n")
#f.write(json.dumps(historique.history))

f.close()


#train_dir =  "/media/sonia/DATA/CASIA_minitrain"
#train_dir ="/media/sonia/DATA/CASIA_SUB_TRAIN/images"
#train_dir =  "/media/sonia/DATA/CASIA_SUB_VAL"
train_dir = "/media/sonia/DATA/CASIA90_TRAIN"


def my_oldmodel():
    vggface = tf.keras.models.Sequential()
    vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME", input_shape=(224,224, 3)))
    vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    vggface.add(tf.keras.layers.Flatten())
    vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
    vggface.add(tf.keras.layers.Dropout(0.5))
    vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
    vggface.add(tf.keras.layers.Dropout(0.5))
    vggface.add(tf.keras.layers.Dense(2622, activation='softmax'))
    #https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5
    vggface.load_weights( 'datas/rcmalli_vggface_tf_vgg16.h5')
    vggface.pop()
    vggface.add(tf.keras.layers.Dense(90,  activation=None))
    vggface.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    print("Architecture custom")
    print( vggface.summary())
    for layer in vggface.layers[:-NOFREEZE]:
        layer.trainable = False
    for layer in  vggface.layers:
        print(layer, layer.trainable)   
    return(vggface)



def vggfacewithoutTOP():
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input, Lambda
    from keras_vggface.vggface import VGGFace
    #custom parameters
    #hidden_dim=512
    hidden_dim = 1024
    #hidden_dim2 = 512
    #hidden_dim3 = 90
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim , activation=None, name = "fc6")(x)
    #x = Dense(hidden_dim1, activation='relu', name='fc6')(x)
    #x = Dense(hidden_dim2, activation='relu', name='fc7')(x)
    #x = Dense(hidden_dim3, activation=None, name='fc8')(x)
    out = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = Model(vgg_model.input, out)
    print("Architecture custom")
    print( model.summary())
    for layer in model.layers[:-2]:
        layer.trainable = False
    for layer in  model.layers:
        print(layer, layer.trainable)   
    return(model)

#--- INPUT PIPELINE WITH TF2

data_dir = train_dir
list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=False)


# get the count of image files in the train directory
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


# To create the single training of validation example with image and its corresponding label
def process_path(file_path):
  label = get_label(file_path)
# load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)


AUTOTUNE = tf.data.experimental.AUTOTUNE
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())



def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(BATCH_SIZE,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


def distance_l2(y1,y2):
    # METTRE COSINE !!!!
     return scipy.spatial.distance.euclidean(y1,y2)**2 ## square of euclidian distance, as in training
 
    
def compute_distance(model,val_ds):
     y = np.concatenate([y for x, y in val_ds], axis=0)
     N = len(y)
     distance_array = np.zeros((N,N))
     vec = model.predict(val_ds,batch_size=BATCH_SIZE,verbose=1)
     for i in range(N):
          for j in range(N):
               d=distance_l2(vec[i][:],vec[j][:])
               distance_array[i,j]=d
     return distance_array


def calc_TOPacc(distance_array, val_ds, TOP = 3):
    y = np.concatenate([y for x, y in val_ds], axis=0)
    TOPacc = []
    #labels = valid_generator.filenames
    for i in range(len(distance_array)):
        L=[]
        #i = 1
        TOP = 3 + 1
        print(TOP)
        distance_i = distance_array[i,:]
        idx = np.argpartition(distance_i, TOP)
        #This returns the k-smallest values. Note that these may not be in sorted order.
        L= list(idx[:TOP])
        print(L)
        print(i)
        if i in L: L.remove(i)
        print(L)
        #L = list(distance_i[idx[:TOP]])
        #print(itemgetter(*L)(labels))
        if y[i] in [y[j] for j in L]:
            TOPacc.append(1)
        else:
            TOPacc.append(0)
    return(sum(TOPacc)/len(TOPacc))


## test compared to train images
class TopX(Callback):
     def __init__(self, x):
          self.x = val_ds
          self.topX = []
          
     def on_train_begin(self, logs={}):
          self.topX = []
          
     def on_epoch_end(self, epoch, logs={}):
        TOP = 3
        distance_array = compute_distance(self.model,self.x)
        #print(self.x.filenames)
        topX = calc_TOPacc(distance_array, self.x, TOP = TOP)
        #print("Top {TOP} - Accuracy(%):   ", round(topX*100,1))
        print(' Top {} - Accuracy {} %'.format(TOP, round(topX*100,1)))
        #print(topX)
        
        
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
vggface = vggfacewithoutTOP()
#vggface = my_oldmodel()
vggface.compile(optimizer=opt, loss=tfa.losses.triplet_semihard_loss )
topx = TopX(val_ds)  
history = vggface.fit(train_ds,  validation_data= val_ds,epochs=NB_EPOCHS,
                      batch_size = BATCH_SIZE,callbacks=[topx])
results = vggface.predict(val_ds)
np.savetxt("results/" + todaystr + "/CFDvecs.tsv", results, delimiter='\t')



sys.stdout.close()


