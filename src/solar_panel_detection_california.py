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


os.chdir('/Users/msabate/Projects/CityFinancial/data/California')
with open('SolarArrayPolygons.geojson') as f:
    data = json.load(f)

def read_geojson(geojson_file):
    """
    this function returns a dictionary with 
    key=image_name
    value=list of polygons coordinates, each polygon being a solar panel
    """
    # example in order to see the polygon_vertices_pixels
    #data['features'][0]['properties']['polygon_vertices_pixels']
    d = dict()
    
    # get all the solar panel coordinates in each picture and save it in a dictionary
    for i in range(len(data['features'])):
        if i%50==0: print(i)
        im_name = data['features'][i]['properties']['image_name']
        if not im_name in d.keys():
            d[im_name] = []
        d[im_name].append(i)
    return(d)


d = read_geojson('SolarArrayPolygons.geojson')

def create_mask(img_id):
    """
    this function creates a mask image with white pixels (1) in the solar panels
    and black pixels (0) everywhere else
    """
    print(img_id)
    img = tiff.imread('images/train/'+img_id+'.tif')
    img_size = img.shape[:2]
    
    img_mask = np.zeros(img_size, np.uint8)
    
    try:
        ind_im = d[img_id]
    except KeyError:
        return(img_id, img, img_mask) # there are no solar panels in that image
    
    #poly = Polygon(data['features'][ind_im[0]]['properties']['polygon_vertices_pixels'])
    polygons=[Polygon(data['features'][i]['properties']['polygon_vertices_pixels']) for i in ind_im]
    int_coords = lambda x: np.rint(np.array(x)).astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [pi.coords for poly in polygons
                     for pi in poly.interiors]
    
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return(img_id, img, img_mask)

    
# example to create and print the mask and the img
img_id = '618651889'
img_id, img, img_mask = create_mask(img_id)
tiff.imshow(img)
mask = 255*np.stack([img_mask, img_mask, img_mask])
tiff.imshow(255*np.stack([img_mask, img_mask, img_mask]))


# list of images in the training set
images = glob('images/train/*.tif')
images_id = [os.path.splitext(os.path.basename(im))[0] for im in images]
data_images = [create_mask(img) for img in images_id]
img = [x[1] for x in data_images]
masks = [x[2] for x in data_images]


# example
tiff.imshow(255*np.stack([masks[0], masks[0], masks[0]]))
#tiff.imsave('images/segmentation/test.png', masks[i])
a = np.where(masks[0]==1)
len(a)



# save the label images with three channels
for i in range(len(images_id)):
    tiff.imsave('images/segmentation/'+images_id[i]+'.png',255*np.stack([masks[i], masks[i], masks[i]]))
    

# save the images with only one channel for image segmentation.
# each array position has a 1 if it belongs to a solar panel
# or a 0 if it is background
for i in range(len(images_id)):
    tiff.imsave('images/segmentation/'+images_id[i]+'.png',255*np.stack([masks[i], masks[i], masks[i]]))
    

# create training set and testing set for images with image size 5000x5000
im5000_name = []
label5000_name = []
im5000 = []
label5000 = []

for x in data_images:
    img_id, img, mask = x
    if mask.shape==(5000,5000): 
        print(img_id)
        tiff.imsave('images/image5000/'+img_id+'.tif', mask)
        tiff.imsave('images/label5000/'+img_id+'.png', img)
        label5000_name.append(img_id+'.png')
        im5000_name.append(img_id+'.tif')
        #label5000.append(mask)
        #im5000.append(img)

train_test = np.random.binomial(n=1, p=0.5, size=len(im5000_name))

# create csv file with two columns:
# first column has image path
# second column has image labels
df_train5000 = pd.DataFrame(data={'im':np.asarray(im5000_name)[np.where(train_test==1)],'label':np.asarray(label5000_name)[np.where(train_test==1)]})
df_train5000.to_csv('images/train5000.csv', header=False, index = False)        

df_test5000 = pd.DataFrame(data={'im':np.asarray(im5000_name)[np.where(train_test==0)],'label':np.asarray(label5000_name)[np.where(train_test==0)]})
df_test5000.to_csv('images/test5000.csv', header=False, index=False)        


    



    
    
    