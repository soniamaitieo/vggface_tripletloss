#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:26:51 2021

@author: sonia
"""

#--- LIBRARIES ---#
import tensorflow as tf
import tensorflow_addons as tfa

import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import sys
import matplotlib.pyplot as plt
from keras.callbacks import Callback

import os
import pathlib
import numpy as np
import random as rd
import pandas as pd
import sklearn.metrics.pairwise 
from scipy.stats import multivariate_normal
import scipy



#--- GLOBAL VAR ---#
img_height, img_width = 224,224
BATCH_SIZE = 128
inputShape = (224, 224)

#tl ind
data_dir = "/home/sonia/cfd/CFDVersion2.5/Images/CFD"
#res_dir = "/home/sonia/vggface_tripletloss/results/2021-02-26-16-57"
#res_dir="/home/sonia/vggface_tripletloss/results/2021-03-09-16-19"
#res_dir = "/home/sonia/vggface_tripletloss/results/2021-03-11-9-54"
res_dir= "/home/sonia/vggface_tripletloss/results/2021-03-12-8-47"
res_dir="/home/sonia/vggface_tripletloss/results/2021-03-04-17-13"

#tl sex
res_dir = "/home/sonia/vggface_tripletloss/results/2021-03-17-17-16"
#--- MODEL ---#

mymodel = tf.keras.models.load_model(res_dir + "/my_model", custom_objects = { 'Loss': tfa.losses.TripletSemiHardLoss },compile=False)


#--- DATA PROCESSING GENDER ---#


imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if 'N.jpg' in val]
cat = [i.split('/')[-2][1] for i in imgs]
list_ds = tf.data.Dataset.from_tensor_slices((imgs, cat))

#--- DATA PROCESSING ---#

#list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*N.jpg'), shuffle=False)
#list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=False)
#list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*.jpg'), shuffle=False)


# get the count of image files in the test directory
image_count= len([i for i in list_ds ])

#get all images files
image_files = [i.numpy().decode("utf-8") for i in list_ds ]
image_files = [i.split('/')[8] for i in image_files ]

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




def process_gender(filename, label):
    #filename = tf.read_file(filename)
    img = tf.io.read_file(filename)  # load the raw data from the file as a string
    img = decode_img(img)
    return img, label

list_ds = list_ds.map(process_gender)
test_ds = list_ds


#--- DATA INCREASE PERF---#

AUTOTUNE = tf.data.experimental.AUTOTUNE
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
  ds = ds.cache()
  #ds = ds.shuffle(buffer_size=1000)
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

test_ds = configure_for_performance(test_ds)

#ind
y = np.concatenate([y for x, y in test_ds], axis=0)
res_emb = mymodel.predict(test_ds)
label = class_names[y]
#y_label = list(y)
y_label = list(label)

#sexe
y_label = [str(i, 'utf-8') for i in y_label]
df_files = pd.DataFrame({'Model': [i.split('/')[-2] for i in imgs],
                         'Image':[i.split('/')[-1] for i in imgs],
                         'y_label':y_label})


#df_files = pd.DataFrame({'Model':label,'Image':image_files, 'y_label':y_label})



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

y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition

# TOP = 5 pour calc mAP@5 
all1000_mAP = []
for it in range(1000) : 
    #On applique pour chaque indiv le calcule de l' average precision
    #mAP_each_ind liste avec le AP de tous les indivdus
    mAP_each_ind= [calc_ap_per_ind(res_emb, i, y_dict,TOP=1) for i in list(y_dict.keys())]
    all1000_mAP.append( np.mean(mAP_each_ind))
    
np.mean(all1000_mAP)

print(" mAP@5 ")
print(np.mean(all1000_mAP))

"""
#np.savetxt(res_dir + "/CFDlabel.tsv", y_label, delimiter='\t')
#np.savetxt(res_dir + "/CFDvecs.tsv", res_emb, delimiter='\t')


take5_index = [x for x in range(len(y_label)) if y_label[x] in rangse(110,110+10)]
res_val5 = res_emb[take5_index]

y_lab5 =[y_label[x] for x in range(len(y_label)) if y_label[x] in range(110,110+10)]
y_lab5_ethn = [x[0] for x in list(class_names[y_lab5]) ]
y_lab5_gender = [x[1] for x in list(class_names[y_lab5]) ]


np.savetxt("CFDlabel10.tsv", y_lab5, delimiter='\t')
np.savetxt("CFDvecs10.tsv", res_val5, delimiter='\t')
np.savetxt("CFDlabel10_gender.tsv", y_lab5_gender, delimiter='\t' ,  fmt='%s')
np.savetxt("CFDlabel1_ethn.tsv", y_lab5_ethn, delimiter='\t')
"""


df = pd.read_excel('/home/sonia/cfd/CFDVersion2.5/CFD2.5NormingDataandCodebook.xlsx',
                   sheet_name='CFD_U.S._NormingData', engine='openpyxl')


df = df.drop([0,1,2,3,4,5])
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(6))

#SELECT
df = df[['Model' , 'GenderSelf', 'Feminine' , 'Attractive' ]]

df_all = df_files.merge(df, how='left',  sort=False)



"""
label = list(np.loadtxt('/home/sonia/vggface_tripletloss/results/2021-03-03-15-33/CFDlabel_testCFD.tsv',delimiter='\t'))
label = [round(num) for num in label]
label = class_names[label]


res_emb = np.loadtxt('/home/sonia/vggface_tripletloss/results/2021-03-03-15-33/CFDvecs_testCFD.tsv',delimiter='\t') 


"""

"""

#--- READ CFD AND EXTRACT INFORMATIVE COL ---#
df = pd.readtxt('/home/sonia/cfd/CFDVersion2.5/CFD2.5NormingDataandCodebook.xlsx',
                   sheet_name='CFD_U.S._NormingData', engine='openpyxl')
df = df.drop([0,1,2,3,4,5])
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(6))

df = df[['Model' , 'GenderSelf', 'Feminine' , 'Attractive' ]]
df = df.dropna()
"""
"""
def calc_LL(res_emb,df):
    L = []
    for i in df.Model :
        print(i)
        pos_i = np.where(label == i)[0][0]
        emb_i = res_emb[[pos_i], :]
        emb_without_i = np.delete(res_emb, (pos_i), axis=0)
        MU = np.mean(emb_without_i, axis=0)
        SIGMA = np.cov(emb_without_i, rowvar=0)                      
        var = multivariate_normal(MU, SIGMA , allow_singular=True )
        #var = multivariate_normal(MU, SIGMA  )
        pdftest=var.pdf(emb_i)                           
        log_pdftest= np.log(pdftest)
        #print(log_pdftest)
        L.append(log_pdftest)
    return(L)

L =  calc_LL(res_emb,df_all)
"""
def calc_LL(res_emb,df):
    L = []
    for i in df.Model :
        print(i)
        pos_i = list(df[df.Model == i].index)
        emb_i = res_emb[pos_i, :]
        emb_without_i = np.delete(res_emb, (pos_i), axis=0)
        MU = np.mean(emb_without_i, axis=0)
        SIGMA = np.cov(emb_without_i, rowvar=0)                      
        #var = multivariate_normal(MU, SIGMA , allow_singular=True )
        var = multivariate_normal(MU, SIGMA  )
        pdftest=var.pdf(emb_i)                           
        log_pdftest= np.log(pdftest)
        #print(log_pdftest)
        L.append(log_pdftest)
    return(L)
    

L = calc_LL(res_emb,df_all)
df_all["LL"] = L


import matplotlib.pyplot as plt
plt.scatter(df_all['Attractive'], df_all['LL'])
from scipy import stats
corr = stats.pearsonr(df_all['Attractive'], df_all['LL'])


def is_normaldist(res_emb): 
    P = []
    from scipy import stats
    for dim in range (128):
        plt.hist(res_emb[:, dim]) 
        k2, p = stats.normaltest(res_emb[:, dim])
        P.append(p)
    toto = [o>0.05 for o in P]
    return(sum(toto))


#df_all = all_df.reset_index()


df_F = df_all[df_all['GenderSelf'] == "F"]
res_emb_F = res_emb[df_F.index]
df_F = df_F.reset_index()

LL_F = calc_LL(res_emb_F,df_F)
df_F['LL_F'] = LL_F



plt.scatter(df_F['Attractive'], df_F['LL_F'])
corr = stats.pearsonr(df_F['Attractive'], df_F['LL_F'])

df_M = df_all[df_all['GenderSelf'] == "M"]
res_emb_M = res_emb[df_M.index]
df_M = df_M.reset_index()
LL_M = calc_LL(res_emb_M,df_M)
df_M['LL_M'] = LL_M
plt.scatter(df_M['Attractive'], df_M['LL_M'])
corr = stats.pearsonr(df_M['Attractive'], df_M['LL_M'])


        
        
df_all = pd.merge(df_all,df_F[['Model','LL_F']], how='left' ,on='Model')
df_all =  pd.merge(df_all,df_M[['Model','LL_M']], how='left' ,on='Model')


fem_moy = np.mean(res_emb_F, axis=0)
men_moy = np.mean(res_emb_M, axis=0)


df_F["dsit_centre_F"] = [ scipy.spatial.distance.euclidean(res_emb_F[i,: ],fem_moy) for i in range(len(res_emb_F))]
df_F["dsit_centre_M"] = [ scipy.spatial.distance.euclidean(res_emb_F[i,: ],men_moy) for i in range(len(res_emb_F))]
plt.scatter(df_F['Feminine'], df_F['dsit_centre_F'])
stats.pearsonr(df_F['Feminine'], df_F['dsit_centre_F'])

plt.scatter(df_F['Feminine'], df_F['dsit_centre_M'])
stats.pearsonr(df_F['Feminine'], df_F['dsit_centre_M'])


df_all["dist_centre_F"] = [ scipy.spatial.distance.euclidean(res_emb[i,: ],fem_moy) for i in range(len(res_emb))]
stats.pearsonr(df_all['Feminine'], df_all["dist_centre_F"])
plt.scatter(df_all['Feminine'], df_all["dist_centre_F"])

df_all["dist_centre_H"] = [ scipy.spatial.distance.euclidean(res_emb[i,: ],men_moy) for i in range(len(res_emb))]
stats.pearsonr(df_all['Feminine'], df_all["dist_centre_H"])
plt.scatter(df_all['Feminine'], df_all["dist_centre_H"])


#df_all.to_csv(res_dir + "/CFD_N_analysis.csv")

"""
np.savetxt(res_dir + "/CFD_LL.tsv", df_all['LL'], delimiter='\t')

np.savetxt(res_dir + "/CFD_Model.tsv", df_all['Model'], delimiter='\t',fmt='%s')

"""

from sklearn import svm

C = 1.0 # paramètre de régularisation
lin_svc = svm.LinearSVC(C=C)
lin_svc = svm.SVC(kernel='linear',C=C,probability=True)



proba_svm = []
predict_svm = []

for i in df_all.Model :
    print(i)
    pos_i = list(df_all[df_all.Model == i].index)
    emb_i = res_emb[pos_i, :]
    emb_without_i = np.delete(res_emb, (pos_i), axis=0)
    gender_i = df_all.GenderSelf[pos_i]
    all_gender = list(df_all.GenderSelf)
    del all_gender[pos_i[0]]
    lin_svc.fit(emb_without_i , all_gender )
    predict_svm.append(lin_svc.predict(emb_i)[0]) 
    proba_svm.append(lin_svc.predict_proba(emb_i)[0][0])
    
true_gender = list(df_all.GenderSelf)
#df_all["dist_centre_H"] = [ scipy.spatial.distance.euclidean(res_emb[i,: ],men_moy) for i in range(len(res_emb))]
res = [idx for idx, elem in enumerate(true_gender)
                           if elem == predict_svm[idx]]


df_all['pF_svm_linear'] = proba_svm 
df_all['Gender_svm_linear_pred'] = predict_svm 




from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(res_emb)
