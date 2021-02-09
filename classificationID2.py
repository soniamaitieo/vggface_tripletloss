#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:32:07 2021

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
data_dir = "/home/sonia/CASIA_minitrain/CASIA_minitrain"
BATCH_SIZE = 32
img_height,img_width = 224,224
inputShape = (224, 224)
LR = 0.001
opt = tf.keras.optimizers.Adam(learning_rate= LR)
NOFREEZE = 4
NB_EPOCHS = 3

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
f.write("base:" + "VGGface with TOP" + "\n")
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


# Labels (numpy) - dim:(nb,)
class_names = [i.split('/')[-2] for i in listpict]
class_names = LabelEncoder().fit_transform(class_names)

#--- DATA TRANSFORM TO TENSOR---#
dataset = tf.data.Dataset.from_tensor_slices((img_data,class_names))


#--- DATA SPLIT---#
val_size = int(len(class_names) * 0.2)
train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


#--- DATA INCREASE PERF---#

AUTOTUNE = tf.data.experimental.AUTOTUNE

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(BATCH_SIZE,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


#--- MODEL---#

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
    for layer in model.layers[:-NOFREEZE]:
        layer.trainable = False
    for layer in model.layers:
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


#--- MAIN---#

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
  batch_size = BATCH_SIZE
)



plot_loss_acc(hist , todaystr)

modeltosave = "results/" + todaystr + "/my_model"
model.save(modeltosave)


end = time.time()
print("Time consuming in (s)")
print(end - start)