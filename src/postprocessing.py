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
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt




os.chdir('/Users/msabate/Projects/CityFinancial/data/California/')

pix_prob = [os.path.basename(x) for x in glob('images/pix_probabilities/*.png')]
pred = [os.path.basename(x) for x in glob('images/pix_probabilities/*.png')]

img_id = [x[6:] for x in pix_prob]

#img_prob = tiff.imread('images/pix_probabilities/'+pix_prob[ind])


l_probs = [tiff.imread('images/pix_probabilities/'+'probs_'+x) for x in img_id[:800]]
l_actual = [tiff.imread('images/label_cropped256/'+x) for x in img_id[:800]]
l_predicted = [tiff.imread('images/prediction/'+'pred_'+x) for x in img_id[:800]]

l_probs_col = [x.reshape(-1) for x in l_probs]
l_actual_col = [x.reshape(-1) for x in l_actual]
l_predicted_col = [x.reshape(-1) for x in l_predicted]

y_score = np.concatenate(l_probs_col, axis=0)
y_true = np.concatenate(l_actual_col, axis=0)
y_predicted = np.concatenate(l_predicted_col, axis=0)


# ROC curve
roc = roc_curve(y_true, y_score)

fpr, tpr, thresholds = roc

plt.figure(1)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr)
plt.show()


# get point on the curve that is closer to [0,1]
distances = np.array([(fpr[i])**2+(1-tpr[i])**2 for i in range(fpr.shape[0])])
t = thresholds[np.argmin(distances)]
t

fpr[np.argmin(distances)]

tpr[np.argmin(distances)]

# Precision recall curve
prec, recall, thresholds2 = precision_recall_curve(y_true, y_score) 

plt.clf()
plt.figure(2)
plt.plot([0,1],[0,1], 'k--')
plt.plot(recall, prec)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()


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
    original_image_PV = cv2.drawContours(original_image, contours_filtered,  -1, (0,255,0), 2)
    #tiff.imshow(original_image_PV)
    original_image = tiff.imread('images/images_cropped256/'+img_name+'.tif')
    
    return original_image, img_actual, img_predicted, original_image_PV

img_id[:10]

original_image, img_actual, img_predicted, original_image_PV = getSolarPanel('10sfg495955_0_9')
tiff.imshow(original_image)
tiff.imshow(img_actual)
tiff.imshow(img_predicted)
tiff.imshow(original_image_PV)

