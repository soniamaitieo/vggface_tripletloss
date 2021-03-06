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

#--- GLOBAL VAR ---#
inputShape = (224, 224)
#num_classes = 1
NOFREEZE = 2
BATCH_SIZE = 128
NB_EPOCHS = 10
#NB_EPOCHS2 = 5
LR = 0.001
opt = tf.keras.optimizers.Adam(learning_rate= LR)


#--- DATASET DIR ---#

#train_dir = '/media/sonia/DATA/CASIA-WebFace_SUB'
#train_dir = "/media/sonia/DATA/casia_sub_sub/images"
train_dir = '/media/sonia/DATA/CASIA_SUB_TRAIN/images'
val_dir = "/media/sonia/DATA/CASIA_SUB_VAL"
test_dir = "/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3Images"

#train_dir = "/media/sonia/DATA/CASIA_minitrain"
#val_dir = "/media/sonia/DATA/CASIA_mini"

#nb img/ind
#find . -maxdepth 1 -mindepth 1 -type d | while read dir; do   printf "%-25.25s : " "$dir";   find "$dir" -type f | wc -l; done
    

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


#--- Functions to calculate our own accuracy ---#

#src: https://plmlab.math.cnrs.fr/vmiele/animal-reid/-/blob/master/deepMetricLearning.py

def distance_l2(y1,y2):
     return scipy.spatial.distance.euclidean(y1,y2)**2 ## square of euclidian distance, as in training
 
    
def compute_distance(model,x):
     N = len(x.filenames)
     distance_array = np.zeros((N,N))
     vec = model.predict(valid_generator,batch_size=16,verbose=1)
     for i in range(N):
          for j in range(N):
               d=distance_l2(vec[i][:],vec[j][:])
               distance_array[i,j]=d
     return distance_array


def calc_TOPacc(distance_array, valid_generator, TOP = 3):
    TOPacc = []
    labels = valid_generator.filenames
    for i in range(len(distance_array)):
        #i = 1
        TOP = 3 + 1
        distance_i = distance_array[i,:]
        original = labels[i]
        idx = np.argpartition(distance_i, TOP)
        #This returns the k-smallest values. Note that these may not be in sorted order.
        L= list(idx[:TOP])
        if i in L: L.remove(i)
        #L = list(distance_i[idx[:TOP]])
        #print(itemgetter(*L)(labels))
        if original.split('/')[0] in [i.split('/', 1)[0] for i in itemgetter(*L)(labels)]:
            TOPacc.append(1)
        else:
            TOPacc.append(0)
    return(sum(TOPacc)/len(TOPacc))
    

## test compared to train images
class TopX(Callback):
     def __init__(self, x):
          self.x = valid_generator
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

with tf.device('/gpu:0'):
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
    vggface.compile(optimizer=opt, loss=tfa.losses.triplet_semihard_loss ,metrics=['accuracy'])
    #vggface.compile( optimizer=opt, loss = tfa.losses.TripletHardLoss())
    #vggface.compile(optimizer=opt, loss=tfa.losses.TripletSemiHardLoss(),metrics=['accuracy'])
    #data_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    #train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)
    data_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255. ,
                                                                     horizontal_flip=False,
                                                                     rotation_range=20,
                                                                     width_shift_range=0.05,
                                                                     height_shift_range=0.05,
                                                                     brightness_range=[0.8,1.2], 
                                                                     shear_range=0.1,
                                                                     zoom_range=[0.85,1.15])
    data_gen_valid = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    data_gen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    train_generator = data_gen_train.flow_from_directory(train_dir, target_size=inputShape, batch_size=BATCH_SIZE, class_mode="sparse")
    valid_generator = data_gen_valid.flow_from_directory(val_dir, target_size=(224,224), batch_size=BATCH_SIZE, class_mode="sparse")
    #history = vggface.fit(train_generator, epochs=NB_EPOCHS , batch_size = BATCH_SIZE)
    topx = TopX(valid_generator)    
    history = vggface.fit_generator(train_generator,epochs=NB_EPOCHS ,callbacks=[topx])
    test_generator = data_gen_valid.flow_from_directory(test_dir, target_size=(224,224), batch_size=BATCH_SIZE, class_mode="sparse")
    results = vggface.predict(test_generator)
    
np.savetxt("results/" + todaystr + "/CFDvecs.tsv", results, delimiter='\t')

#import io 

#out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#for i in valid_generator:
    #[out_m.write(str(x) + "\n") for x in valid_generator.filenames]
#out_m.close()



#------------------------------------------------------------



MU = np.mean(results, axis=0)
SIGMA = np.cov(results, rowvar=0)
                           
from scipy.stats import multivariate_normal
var = multivariate_normal(MU, SIGMA)
pdftest=var.pdf(results)                           
log_pdftest= np.log(pdftest)
np.savetxt("results/" + todaystr + "/CFD_LL.tsv", log_pdftest, delimiter='\t')






