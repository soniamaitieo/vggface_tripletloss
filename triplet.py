#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:52:34 2021

@author: renoult
"""
import sys
import io
import os
import pathlib
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda
from keras_vggface.vggface import VGGFace
import sklearn.metrics.pairwise
from keras.callbacks import Callback
import random as rd
"""
# load_ext tensorboard
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=513802240)])
  except RuntimeError as e:
    print(e)

"""

img_height = 224
img_width = 224
batch_size = 128
n_epochs = 50
LR = 0.001
LR2 = 0.00001
NOFREEZE = 5
dr = 0.5

#data_dir = "/media/sonia/DATA/CASIA90_TRAIN"
#data_dir ="/media/sonia/DATA/CASIA_SUB_TRAIN/images"
#data_dir =  "/media/sonia/DATA/CASIA_minitrain"
data_dir = "/media/sonia/DATA/facescrub/download/faces"
data_dir = pathlib.Path(data_dir)
#image_count = len(list(data_dir.glob('*/*.jpg')))
image_count = len(list(data_dir.glob('*/*.jpeg')))

print(image_count)


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
f.write("base:" + "VGGface VGG16 TOP false" + "\n")
f.write("weights:" + "VGGface " + "\n")
f.write("epochs:" + str(n_epochs) + "\n")
#f.write("epochs2:" + str(NB_EPOCHS2) + "\n")
f.write("batch_size:" + str(batch_size) + "\n" )
f.write("optim1:" + 'Adam' + "\n")
f.write("optim1_LR:" + str(LR) + "\n")
f.write("optim2_LR:" + str(LR2) + "\n")
#f.write("no freezen layer:" + str(NOFREEZE) + "\n")
f.write("dataset_dir:" + str(data_dir) + "\n")
#f.write("Img_preprocessing" + "rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)")
#f.write("Img_preprocessing:" + "rescale=1./255, horizontal_flip=False, rotation_range=20,width_shift_range=0.05, height_shift_range=0.05,brightness_range=[0.8,1.2], shear_range=0.1,zoom_range=[0.85,1.15]")
#f.write("val:" + str(0.20) + "\n")
#f.write("early_stop:" + str(10) + "\n")
#f.write("reducelr0.1:" + str(5) + "\n")
#f.write("dropout:" + str(dr) + "\n")
#f.write("HorizontalFlip:" + "True" + "\n")
#f.write(json.dumps(historique.history))
f.close()



#--- DATA PROCESSING ---#
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpeg'), shuffle=False)
#list_ds = tf.data.Dataset.list_files("/media/sonia/DATA/CASIA_SUB_TRAIN/images" + '/*/*', shuffle=False) #shuffle ????
#list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))
print(class_names)
"""

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)


"""


"""
train_size = int(0.7 * image_count)
val_size = int(0.1 * image_count)
test_size = int(0.2 * image_count)

train_ds = list_ds.take(train_size)
test_ds = list_ds.skip(train_size)
val_ds = test_ds.skip(val_size)
test_ds = test_ds.take(test_size)

"""



#print(tf.data.experimental.cardinality(train_ds).numpy())
#print(tf.data.experimental.cardinality(val_ds).numpy())
#print(tf.data.experimental.cardinality(test_ds).numpy())

#Convert path to (img, label) tuple
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)  # convert the path to a list of path components
  one_hot = parts[-2] == class_names  # The second to last is the class-directory
  return tf.argmax(tf.cast(one_hot, tf.int32))
"""
def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)   # convert the compressed string to a 3D uint8 tensor
  img = tf.cast(img, tf.float32) / 255.
  return tf.image.resize(img, [img_height, img_width])   # convert the compressed string to a 3D uint8 tensor
"""
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

AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
"""


train_ds = list_ds.take(n_train)
val_ds = list_ds.skip(n_train).take(n_valid)
test_ds = list_ds.skip(n_train + n_valid).take(n_test)
"""
#CLOSE-DS#
# Size of dataset
n = sum(1 for _ in list_ds)
n_train = int(n * 0.7)
n_val = int(n * 0.1)
#n_test= int(n * 0.01)
n_test = n - n_train - n_val #0.2

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
    
    
train_ds,val_ds,test_ds=choose_ds(list_ds , "semiclose", image_count,n_train,n_val,n_test)  
  
def augment0(image, label):
    #en radian 0.5 environ 30°
    #img = tfa.image.transform_ops.rotate(image, 0.5)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    #img = tf.image.random_brightness(img, 0.3)
    # Center cropping of the image (the cropping area is at the center)
    rd_angle = round(rd.uniform(0, 0.2), 2)
    img = tfa.image.transform_ops.rotate(image, rd_angle)
    #central_fraction = 0.6 # The scale of the cropped area to the original image
    #img = tf.image.central_crop(img, central_fraction=central_fraction)
     # Resize the image to add four extra pixels on each side.
    #img = tf.image.resize_image_with_crop_or_pad(img, img_height + 8, img_width + 8)
    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    #img = tf.random_crop(image, [img_height,img_width, 3])
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    #central_fraction = 0.6 # The scale of the cropped area to the original image
    #img = tf.image.central_crop(img, central_fraction=central_fraction)
    #img = tf.image.random_crop(img, size=[img_height,img_width, 3])
    return (img, label)

def augment(image, label):
    rd_angle = round(rd.uniform(0, 0.2), 2)
    img = tfa.image.transform_ops.rotate(image, rd_angle)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, 0.3)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    return (img, label)


def augmentso(image, label):
    #en radian 0.5 environ 30°
    img = tfa.image.transform_ops.rotate(image, 0.5)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_brightness(img, 0.3)
    return (img, label)


# optimze preprocessing performance
def configure_for_performance(ds):
  ds = ds.cache()
  #ds = ds.shuffle(buffer_size=n_train)
  ds = ds.map(augment, num_parallel_calls=AUTOTUNE) # augmentation call
  ds = ds.batch(batch_size,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
#val_ds = configure_for_performance(val_ds)
val_ds = val_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE) #retirer pour 518
test_ds =test_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE) #retirer pour 518

"""
#--- DATA PREPROCESS TEST ---#
data_dir_test = "/media/sonia/DATA/CASIA90_VAL"
#data_dir_test = "/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3Images"
#--- DATA PROCESSING ---#
#list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*-N.jpg'), shuffle=False)
list_ds_test = tf.data.Dataset.list_files(str(data_dir_test + '/*/*.jpg'), shuffle=False)
image_count= len([i for i in list_ds_test ])
class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir_test)]))
test_ds = list_ds_test
AUTOTUNE = tf.data.experimental.AUTOTUNE

test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)


#--- DATA PREPROCESS TEST cfd ---#
#data_dir_test = "/media/sonia/DATA/CASIA90_VAL"
data_dir_CFD = "/media/sonia/DATA/CFDVersion2.0.3/CFD2.0.3Images"
#--- DATA PROCESSING ---#
#list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*-N.jpg'), shuffle=False)
list_ds_CFD = tf.data.Dataset.list_files(str(data_dir_CFD + '/*/*.jpg'), shuffle=False)
image_count= len([i for i in list_ds_CFD ])
class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir_CFD)]))
test_ds_CFD= list_ds_CFD
AUTOTUNE = tf.data.experimental.AUTOTUNE

test_ds_CFD = test_ds_CFD.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds_CFD = test_ds_CFD.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
vgg_model = vgg_model.layers.pop()
model = tf.keras.models.Sequential(vgg_model.layers[:-2]) #vgg16
model.add(tf.keras.layers.Dense(128, activation=None))
model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
model.summary()
for layer in model.layers[:-4]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)

#----------------
vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = tf.keras.layers.Dense(128, activation=None)(x)
out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
model = Model(vgg_model.input, out)
for layer in model.layers[:-3]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)


vgg_model = VGGFace(model='resnet50', input_shape=(224, 224, 3) , include_top=True)
last_layer = vgg_model.layers[-1].output
x = tf.keras.layers.Dense(128, activation=None)(last_layer)
out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
model = Model(vgg_model.input, out)
for layer in model.layers[:-4]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)


vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = tf.keras.layers.Dense(128, activation=None)(x)
out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
model = Model(vgg_model.input, out)
for layer in model.layers[:-3]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)


###VGG MODEL###


"""

def model_resnet50_2():
    from keras_vggface.vggface import VGGFace
    from keras.engine import  Model
    vgg_model = VGGFace(model='resnet50', input_shape=(224, 224, 3) , include_top=True)
    last_layer = vgg_model.layers[-2].output
    x = tf.keras.layers.Dropout(0.2)(last_layer)
    x = tf.keras.layers.Dense(128 ,activation=None)(x)
    out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = Model(vgg_model.input, out)
    print("Architecture custom")
    print(model.summary())
    # -8 si defreeze le dernier bloc de conv
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)

    return(model)

def model_vgg16():
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

#model = model_resnet50_2()
model = model_vgg16()

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
        self.TOP = 5
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


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    #optimizer = tf.keras.optimizers.Adam(LR,beta_1=0.02,beta_2=0.02),
    loss=tfa.losses.TripletSemiHardLoss())

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=n_epochs,
   callbacks=[map_N(val_ds),map_N(train_ds)]
    )

mAP_history_val, mAP_history_train = mAP_history[::2],mAP_history[1::2]

def plot_loss_acc(hist , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist.history['loss']
    val_loss = hist.history['val_loss']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.legend(['train','val'],loc=4)
    #plt.title('train_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss')
    plt.grid(True)
    plt.legend(['train'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    #plt.plot(xc, mAP_history)
    plt.plot(xc, mAP_history_train)
    plt.plot(xc, mAP_history_val)
    plt.xlabel('num of Epochs')
    plt.ylabel('mAP@5 accuracy')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()

plot_loss_acc(history , todaystr)

"""
y_label = list(np.concatenate([y for x, y in test_ds], axis=0))
res_val = model.predict(test_ds,verbose=1) #embedding

y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition

#TOP = 5 pour calc mAP@5 
all1000_mAP = []
for it in range(100) : 
    #On applique pour chaque indiv le calcule de l' average precision
    #mAP_each_ind liste avec le AP de tous les indivdus
    mAP_each_ind= [calc_ap_per_ind(res_val,i,  y_dict,TOP=5) for i in list(y_dict.keys())]
    all1000_mAP.append( np.mean(mAP_each_ind))
    
np.mean(all1000_mAP)

print(" mAP@5 ")
print(np.mean(all1000_mAP))
"""
modeltosave = "results/" + todaystr + "/my_model"
model.save(modeltosave)

"""
y_label = list(np.concatenate([y for x, y in train_ds], axis=0))
y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
mAP_each_ind= [calc_ap_per_ind(model.predict(train_ds,verbose=1),i,  y_dict,TOP=5) for i in list(y_dict.keys())]
"""
y_label = list(np.concatenate([y for x, y in test_ds], axis=0))
res_val = model.predict(test_ds,verbose=1) #embedding
y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition

np.savetxt(("results/"   +  todaystr  + "/label_test.tsv"), y_label, delimiter='\t')
np.savetxt(("results/"   +  todaystr  + "/vecs_test.tsv"), res_val , delimiter='\t')


all1000_mAP = []
for it in range(100) : 
    #On applique pour chaque indiv le calcule de l' average precision
    #mAP_each_ind liste avec le AP de tous les indivdus
    mAP_each_ind= [calc_ap_per_ind(res_val,i,  y_dict,TOP=5) for i in list(y_dict.keys())]
    all1000_mAP.append( np.mean(mAP_each_ind))
    
np.mean(all1000_mAP)

print(" mAP@5 (IT100) - TEST ")
print(np.mean(all1000_mAP))

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




res_val5 = res_val[[x for x in range(len(y_label)) if y_label[x] in range(10)]]
y_label5 = [ y_label[x] for x in range(len(y_label)) if y_label[x] in range(10)]

#res_val5 = res_val[[x for x in range(len(y_label)) if y_label[x] in range(109,109+158)]]
#y_label5 = [ y_label[x] for x in range(len(y_label)) if y_label[x] in range(109,109+158)]

np.savetxt(("results/"   +  todaystr  + "/label_test_10.tsv"), y_label5, delimiter='\t')
np.savetxt(("results/"   +  todaystr  + "/vecs_test_10.tsv"), res_val5 , delimiter='\t')

