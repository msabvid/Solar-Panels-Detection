#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:32:01 2017

@author: msabate
"""



#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:41:34 2017

@author: msabate
"""

import json
import os
from shapely.geometry import shape, Polygon
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import sklearn as sk
from glob import glob
from skimage.filters.rank import entropy
from skimage.morphology import rectangle 
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from functools import reduce
import seaborn as sns
import gc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt




os.chdir('/Users/msabate/Projects/CityFinancial/data/California/')

pix_prob = [os.path.basename(x) for x in glob('images/pix_probabilities2/*.png')]
pred = [os.path.basename(x) for x in glob('images/pix_probabilities2/*.png')]

img_id = [x[6:] for x in pix_prob]

#img_prob = tiff.imread('images/pix_probabilities/'+pix_prob[ind])


l_probs = [tiff.imread('images/pix_probabilities2/'+'probs_'+x) for x in img_id[:900]]
l_actual = [tiff.imread('images/label_cropped256/'+x) for x in img_id[:900]]
l_predicted = [tiff.imread('images/prediction2/'+'pred_'+x) for x in img_id[:900]]

l_probs_col = [x.reshape(-1) for x in l_probs]
l_actual_col = [x.reshape(-1) for x in l_actual]
l_predicted_col = [x.reshape(-1) for x in l_predicted]

y_score = np.concatenate(l_probs_col, axis=0)
y_true = np.concatenate(l_actual_col, axis=0)
y_predicted = np.concatenate(l_predicted_col, axis=0)


precision_score(y_true, y_predicted)
recall_score(y_true, y_predicted)
accuracy_score(y_true, y_predicted)

# ROC curve
roc = roc_curve(y_true, y_score)

fpr, tpr, thresholds = roc

fig = plt.figure(1)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr)
plt.show()
fig.savefig('/Users/msabate/Desktop/ROC.png')



# get point on the curve that is closer to [0,1]
distances = np.array([(fpr[i])**2+(1-tpr[i])**2 for i in range(fpr.shape[0])])
t = thresholds[np.argmin(distances)]
t

fpr[np.argmin(distances)]

tpr[np.argmin(distances)]

# Precision recall curve
prec, recall, thresholds2 = precision_recall_curve(y_true, y_score) 

plt.clf()
fig2 = plt.figure(2)
plt.plot([0,1],[0,1], 'k--')
plt.plot(recall, prec)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()
fig2.savefig('/Users/msabate/Desktop/precision_recall.png')



# Threshold
y_pred_threshold = np.zeros_like(y_score)
y_pred_threshold[y_score>t] = 1

precision_score(y_true, y_pred_threshold)
recall_score(y_true, y_pred_threshold)

precision_score(y_true, y_predicted)
recall_score(y_true, y_predicted)


# work with contours
def create_predicted_from_threshold(threshold, probs):
    predicted_threshold = np.zeros_like(probs)
    predicted_threshold[probs>threshold] = 1
    return predicted_threshold


l_predicted_threshold = [create_predicted_from_threshold(t, x) for x in l_probs]
mylist = l_predicted_threshold
mylist = l_predicted


[np.unique(x) for x in mylist]
img = 255*mylist[4][0,:,:].astype('uint8')
img = np.stack([img, img, img])
tiff.imshow(img)

img_actual = 255*l_actual[4].astype('uint8')
img_actual = np.stack([img_actual, img_actual, img_actual])
tiff.imshow(img_actual)

img = img.transpose(1,2,0)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im2,contours,hierarchy = cv2.findContours(gray_image, 1, 2)

[cv2.contourArea(x) for x in contours]
im2

# Obviously thresholding does not work
def getSolarPanel(img_name):
    ind = img_id.index(img_name+'.png')
    
    img_predicted = 255*l_predicted[ind][0,:,:].astype('uint8')
    img_predicted = np.stack([img_predicted, img_predicted, img_predicted])
    #tiff.imshow(img)
    
    img_actual = 255*l_actual[ind].astype('uint8')
    img_actual = np.stack([img_actual, img_actual, img_actual])
    #tiff.imshow(img_actual)
    
    img_predicted = img_predicted.transpose(1,2,0)
    gray_image = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    im2,contours,hierarchy = cv2.findContours(gray_image, 1, 2)
    
    #[cv2.contourArea(x) for x in contours]
    
    contours_filtered = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>10:
            contours_filtered.append(contours[i])
            
    #[cv2.contourArea(x) for x in contours_filtered]
    
    original_image = tiff.imread('images/images_cropped256/'+img_name+'.tif')
    #tiff.imshow(original_image)
    original_image_PV = cv2.drawContours(original_image, contours_filtered,  -1, (255,0,0), 2)
    #tiff.imshow(original_image_PV)
    original_image = tiff.imread('images/images_cropped256/'+img_name+'.tif')
    
    return original_image, img_actual, img_predicted, original_image_PV

img_id[:10]

for i in range(10):
    img_name = img_id[i].split('.')[0]
    original_image, img_actual, img_predicted, original_image_PV = getSolarPanel(img_name)
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    im1 = ax1.imshow(original_image,interpolation='none')
    im2 = ax2.imshow(img_predicted,interpolation='none')
    im3 = ax3.imshow(original_image_PV,interpolation='none')
    fig.savefig('images/prediction_composition/'+img_name+'.png')



original_image, img_actual, img_predicted, original_image_PV = getSolarPanel('10sfg495955_3_8')
tiff.imshow(original_image)
tiff.imshow(img_actual)
tiff.imshow(img_predicted)
tiff.imshow(original_image_PV)


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
im1 = ax1.imshow(original_image,interpolation='none')
im2 = ax2.imshow(img_predicted,interpolation='none')
im3 = ax3.imshow(original_image_PV,interpolation='none')
fig.show()
#fig.savefig('/Users/msabate/Desktop/Original_Prediction')



tiff.imsave('/Users/msabate/Desktop/original_image1.tif', original_image)
tiff.imsave('/Users/msabate/Desktop/predicted1.jpg', img_predicted)
tiff.imsave('/Users/msabate/Desktop/original_image_PV1.tif', original_image_PV)



original_image, img_actual, img_predicted, original_image_PV = getSolarPanel('10sfg495955_6_3')
tiff.imshow(original_image)
tiff.imshow(img_actual)
tiff.imshow(img_predicted)
tiff.imshow(original_image_PV)

tiff.imsave('/Users/msabate/Desktop/original_image2.tif', original_image)
tiff.imsave('/Users/msabate/Desktop/predicted2.jpg', img_predicted)
tiff.imsave('/Users/msabate/Desktop/original_image_PV2.tif', original_image_PV)


for ind in range(10):
    img_predicted = 255*l_predicted[ind][0,:,:].astype('uint8')
    img_predicted = np.stack([img_predicted, img_predicted, img_predicted])
    tiff.imshow(img_predicted)
    
    
    

# check how many solar panels are correctly predicted
def getSolarPanels_Recall(img_name):
    ind = img_id.index(img_name+'.png')
    
    img_predicted = 255*l_predicted[ind][0,:,:].astype('uint8')
    img_predicted = np.stack([img_predicted, img_predicted, img_predicted])
    #tiff.imshow(img)
    
    img_actual = 255*l_actual[ind].astype('uint8')
    img_actual = np.stack([img_actual, img_actual, img_actual])
    #tiff.imshow(img_actual)
    
    img_predicted = img_predicted.transpose(1,2,0)
    gray_image = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    im2,contours,hierarchy = cv2.findContours(gray_image, 1, 2)
    
    #[cv2.contourArea(x) for x in contours]
    
    contours_filtered = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>10:
            contours_filtered.append(contours[i])
    

    img_actual = img_actual.transpose(1,2,0)
    gray_image_actual = cv2.cvtColor(img_actual, cv2.COLOR_BGR2GRAY)
    img_actual2,contours_actual,hierarchy_actual = cv2.findContours(gray_image_actual, 1, 2)
    
    solar_panel_list = []
    predicted_list = []
    for i in range(len(contours_actual)):

        #d[img_name+'_solarpanel'+str(i)] = 0
        try:
            p1 = Polygon(contours_actual[i].reshape(-1,2))
            solar_panel_list.append(img_name+'_solarpanel'+str(i))
            predicted_list.append(0)
            for j in range(len(contours_filtered)):
                p2 = Polygon(contours_filtered[j].reshape(-1,2))
                if p1.intersects(p2):
                    predicted_list[-1] = 1
                    break
        except:
            print('woh there was a linear solar panel')    
    
    df = pd.DataFrame({'solar_panel': solar_panel_list, 'predicted': predicted_list})        
        
    return pd.DataFrame(df)


def getSolarPanels_Precision(img_name):
    ind = img_id.index(img_name+'.png')
    
    img_predicted = 255*l_predicted[ind][0,:,:].astype('uint8')
    img_predicted = np.stack([img_predicted, img_predicted, img_predicted])
    #tiff.imshow(img)
    
    img_actual = 255*l_actual[ind].astype('uint8')
    img_actual = np.stack([img_actual, img_actual, img_actual])
    #tiff.imshow(img_actual)
    
    img_predicted = img_predicted.transpose(1,2,0)
    gray_image = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    im2,contours,hierarchy = cv2.findContours(gray_image, 1, 2)
    
    #[cv2.contourArea(x) for x in contours]
    
    contours_filtered = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>10:
            contours_filtered.append(contours[i])
    

    img_actual = img_actual.transpose(1,2,0)
    gray_image_actual = cv2.cvtColor(img_actual, cv2.COLOR_BGR2GRAY)
    img_actual2,contours_actual,hierarchy_actual = cv2.findContours(gray_image_actual, 1, 2)
    
    solar_panel_list = []
    predicted_list = []
    for i in range(len(contours_filtered)):

        #d[img_name+'_solarpanel'+str(i)] = 0
        try:
            p1 = Polygon(contours_filtered[i].reshape(-1,2))
            solar_panel_list.append(img_name+'_solarpanel'+str(i))
            predicted_list.append(0)
            for j in range(len(contours_actual)):
                p2 = Polygon(contours_actual[j].reshape(-1,2))
                if p1.intersects(p2):
                    predicted_list[-1] = 1
                    break
        except:
            print('woh there was a linear solar panel')    
    
    df = pd.DataFrame({'solar_panel_predicted': solar_panel_list, 'exists': predicted_list})        
        
    return pd.DataFrame(df)



list_df = []
for i in range(900):
    img_name = img_id[i].split('.')[0]
    #print(i)
    list_df.append(getSolarPanels_Recall(img_name))
    
df_recall = reduce(lambda x,y: pd.concat([x,y]), list_df)    
df_recall.shape
df_recall['predicted'].sum()

recall_solarPanels = df_recall['predicted'].sum()/df_recall.shape[0]


list_df = []
for i in range(900):
    img_name = img_id[i].split('.')[0]
    #print(i)
    list_df.append(getSolarPanels_Precision(img_name))
    
df_precision = reduce(lambda x,y: pd.concat([x,y]), list_df)    
df_precision.shape
df_precision['exists'].sum()

precision_solarPanels = df_precision['exists'].sum()/df_precision.shape[0]






    













