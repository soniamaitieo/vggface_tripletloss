#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:49:02 2021

@author: sonia
"""

#--- LIBRARIES ---#

import tensorflow as tf
import os
import numpy as np
import time
import datetime
import sys
import matplotlib.pyplot as plt
from keras.callbacks import Callback
start = time.time()

#--- GLOBAL VAR ---#
inputShape = (224, 224)
img_height, img_width = 224,224
#num_classes = 1
NOFREEZE = 2
BATCH_SIZE = 128
NB_EPOCHS = 15
#NB_EPOCHS2 = 5
LR = 0.002
opt = tf.keras.optimizers.Adam(learning_rate= LR)
LR2 = 0.00001
opt2 = tf.keras.optimizers.Adam(learning_rate= LR2)



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
f.write("base:" + "VGGface without TOP" + "\n")
f.write("weights:" + "VGGface " + "\n")
f.write("epochs:" + str(NB_EPOCHS) + "\n")
#f.write("epochs2:" + str(NB_EPOCHS2) + "\n")
f.write("batch_size:" + str(BATCH_SIZE) + "\n" )
f.write("optim1:" + str(type(opt))  + "\n")
f.write("optim1_LR:" + str(LR) + "\n")
#f.write("optim2:" + OPTIM2  + "\n")
f.write("no freezen layer:" + str(NOFREEZE) + "\n")
#f.write("Img_preprocessing" + "rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)")
#f.write("Img_preprocessing:" + "rescale=1./255, horizontal_flip=False, rotation_range=20,width_shift_range=0.05, height_shift_range=0.05,brightness_range=[0.8,1.2], shear_range=0.1,zoom_range=[0.85,1.15]")
#f.write("val:" + str(0.20) + "\n")
#f.write("early_stop:" + str(10) + "\n")
#f.write("reducelr0.1:" + str(5) + "\n")
#f.write("dropout:" + str(0.5) + "\n")
#f.write("HorizontalFlip:" + "True" + "\n")
#f.write(json.dumps(historique.history))
f.close()



#-----------------------------------------------------------------------------


#--- DATA DIR ---#

data_dir ="/media/sonia/DATA/CASIA_SUB_TRAIN/images"
#data_dir = "/media/sonia/DATA/CASIA90_TRAIN"

#--- DATA PROCESSING ---#

list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=False) #shuffle ????

# get the count of image files in the directory
image_count=0
for dir1 in os.listdir(data_dir):
    for files in os.listdir(os.path.join(data_dir, dir1)):
        image_count+=1
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))
print(class_names)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


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

#------------------------------------------------------------------------------
#--- MODELS---#

def vggfacewithoutTOP_classif():
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input, Lambda
    from keras_vggface.vggface import VGGFace
    #custom parameters
    #hidden_dim=512
    hidden_dim1 = 1024
    #hidden_dim = len(class_names)
    #hidden_dim2 = 512
    hidden_dim3 = len(class_names)
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    #out = Dense(hidden_dim)(x)
    x = Dense(hidden_dim1, activation='relu', name='fc6')(x)
    #x = Dense(hidden_dim2, activation='relu', name='fc7')(x)
    out = Dense(hidden_dim3, activation='softmax', name='fc8')(x)
    #out = Dense(x, activation=None, name='fc8')(x)
    #out = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = Model(vgg_model.input, out)
    print("Architecture custom")
    print( model.summary())
    for layer in model.layers[:-NOFREEZE]:
        layer.trainable = False
    for layer in  model.layers:
        print(layer, layer.trainable)
    return(model)



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
    vggface.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    #vggface.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    print("Architecture custom")
    print( vggface.summary())
    for layer in vggface.layers[:-NOFREEZE]:
        layer.trainable = False
    for layer in  vggface.layers:
        print(layer, layer.trainable)
    return(vggface)


def model3() :
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input, Lambda
    from keras_vggface.vggface import VGGFace
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(class_names), activation= 'softmax', name='fc8' ))
    #model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', name='fc8' ))
    print("Architecture custom")
    print( model.summary())
    for layer in model.layers[:-1]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)
    
def model4() :
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input, Lambda
    from keras_vggface.vggface import VGGFace
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(90, name='fc8' ,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(class_names), activation= 'softmax', name='fc9' ))
    #model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', name='fc8' ))
    print("Architecture custom")
    print( model.summary())
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model5(model4) :
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input, Lambda
    vgg_model = model4
    vgg_model.pop()
    vgg_model.add(tf.keras.layers.Dense(90, name='fc9' ,activation='relu'))
    vgg_model.add(tf.keras.layers.Dropout(0.5))
    vgg_model.add(tf.keras.layers.Dense(len(class_names), activation= 'softmax', name='fc10' ))
    print("Architecture custom")
    print(vgg_model.summary())
    for layer in vgg_model.layers[:-4]:
        layer.trainable = False
    for layer in vgg_model.layers:
        print(layer, layer.trainable)
    return(model)


#--- TRACKING ---#

logdir = "results/" + todaystr
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#file_writer = tf.summary.create_file_writer(logdir + "/metrics")
#file_writer.set_as_default()

def plot_loss_acc(hist , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['sparse_categorical_accuracy']
    val_acc=hist.history['val_sparse_categorical_accuracy']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()



## DECAY LR
def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.001
  if epoch > 1:
    learning_rate = 0.0001
  if epoch > 5:
    learning_rate = 0.00001
  if epoch > 10:
    learning_rate = 0.000001

  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)



#--- MAIN ---#

#FIRS
#model = my_oldmodel()
#model = vggfacewithoutTOP_classif()
model = model4()

# Compile the model
model.compile(optimizer=opt,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy']

)

hist = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=NB_EPOCHS,
  batch_size = BATCH_SIZE,
  callbacks=[lr_callback]
)

plot_loss_acc(hist , todaystr)

modeltosave = "results/" + todaystr + "/my_model"
model.save(modeltosave)


end = time.time()
print("Time consuming in (s)")
print(end - start)

"""
model4 = model
model_emb = model5(model4)
#Compile the model
model_emb.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.00001),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy']

)

hist2 =model_emb.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5,
  batch_size = BATCH_SIZE
)

#plot_loss_acc(hist2 , todaystr)

#logit : https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
"""
