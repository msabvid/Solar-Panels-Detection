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

gc.collect()

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

def create_mask(img_path):
    """a
    this function creates a mask image with white pixels (1) in the solar panels
    and black pixels (0) everywhere else
    """
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    print(img_id)
    #img = tiff.imread('images/train/'+img_id+'.tif')
    img = tiff.imread(img_path)
    img_size = img.shape[:2]
    
    img_mask = np.zeros(img_size, np.uint8)
    
    try:
        ind_im = d[img_id]
    except KeyError:
        return(img_id, img, img_mask, [], []) # there are no solar panels in that image
    
    #poly = Polygon(data['features'][ind_im[0]]['properties']['polygon_vertices_pixels'])
    polys=[data['features'][i]['properties']['polygon_vertices_pixels'] for i in ind_im]
    lengths = [len(p) for p in polys]
    if 2 in lengths: del polys[lengths.index(2)]
#    
    polygons = [Polygon(p) for p in polys]

    areas = [p.area for p in polygons]
    #polygons=[Polygon(data['features'][i]['properties']['polygon_vertices_pixels']) for i in ind_im]
    int_coords = lambda x: np.rint(np.array(x)).astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [pi.coords for poly in polygons
                     for pi in poly.interiors]
    

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    
    solar_PV = img[img_mask==1,:]
    
    return(img_id, img, img_mask, areas, solar_PV)

    
# example to create and print the mask and the img
img_path = '/Users/msabate/Projects/CityFinancial/data/California/images/images_fresno/11ska625695.tif'
img_id, img, img_mask, _, _ = create_mask(img_path)
tiff.imshow(img)
mask = 255*np.stack([img_mask, img_mask, img_mask])
tiff.imshow(255*np.stack([img_mask, img_mask, img_mask]))


img_path = '/Users/msabate/Projects/CityFinancial/data/California/images/images_oxnard/621051897.tif'
img_id, img, img_mask, _, _ = create_mask(img_path)
tiff.imshow(img)
mask = 255*np.stack([img_mask, img_mask, img_mask])
tiff.imshow(255*np.stack([img_mask, img_mask, img_mask]))






# list of images in the training set
list_dirs = ['images/images_fresno', 'images/images_oxnard', 'images/images_modesto', 'images/images_stockton']   # add other cities if necessary
images = [glob(path+'/*.tif') for path in list_dirs]
images = reduce(lambda x,y: x+y, images)
#images_id = [os.path.splitext(os.path.basename(im))[0] for im in images]
data_images = [create_mask(img) for img in images]
img = [x[1] for x in data_images]
masks = [x[2] for x in data_images]
images_id = [x[0] for x in data_images]
areas = reduce(lambda x,y: x+y, [z[3] for z in data_images]) 
solar_colours = filter(lambda x: type(x)!=list, [x[4] for x in data_images])
solar_colours = reduce(lambda x,y: np.concatenate((x,y),axis=0), solar_colours)
gc.collect()

##################################
# study of areas of solar panels #
##################################
areas_df = pd.DataFrame(areas)
areas_df.describe(percentiles=np.linspace(0,1,num=21))

dist_areas = sns.distplot(np.array(areas)[np.array(areas)<1000], kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
dist_areas.get_figure().savefig('dist_areas.png')


##################################
# study of solar panels colours ##
##################################

solar_colours_df = pd.DataFrame(solar_colours, columns=['R', 'G', 'B'])
solar_colours_df.describe(percentiles=np.linspace(0,1,num=11))

solar_colours_df.hist(bins=500)

sns.jointplot(x='R', y='G', data=solar_colours_df, 
              joint_kws={'s':0.5, 'alpha':0.3})

sns.jointplot(x='R', y='B', data=solar_colours_df, 
              joint_kws={'s':0.5, 'alpha':0.2})

sns.jointplot(x='G', y='B', data=solar_colours_df, 
              joint_kws={'s':0.5, 'alpha':0.2})



gc.collect()

#sns.jointplot(x='R', y='G', data=solar_colours_df, kind="kde")


R = solar_colours_df.R.value_counts().to_frame()
a = pd.DataFrame(data={'ind':range(256)})
R_all = pd.merge(a,R,how='left',left_index=True, right_index=True)
R_all.loc[pd.isnull(R_all.R)]


perc = R.apply(lambda x: x/R.sum())
df_R = pd.DataFrame(dict(count = R, p = perc))


###############
# example
###############
tiff.imshow(255*np.stack([masks[375], masks[375], masks[375]]))
#tiff.imsave('images/segmentation/test.png', masks[i])
a = np.where(masks[0]==1)
len(a)



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
    img_id, img, mask, _, _ = x
    if mask.shape==(5000,5000): 
        print(img_id)
        tiff.imsave('images/image5000/'+img_id+'.tif', img)
        tiff.imsave('images/label5000/'+img_id+'.png', mask)
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



# if we want to crop

def cropper(x, width, height):
    img_id, img, mask, _, _ = x
    im3 = entropy(255*mask, rectangle(width, height))
    try:
        coord_x = max(np.where(im3==np.max(im3))[0][0]-width//2,0)
        coord_y = max(np.where(im3==np.max(im3))[1][0]-height//2,0)
    except:
        coord_x = 0
        coord_y = 0
    if coord_x+width>mask.shape[0]: coord_x = mask.shape[0]-width
    if coord_y+height>mask.shape[1]: coord_y = mask.shape[1]-height
    cropped_label = mask[coord_x:coord_x+width, coord_y:coord_y+height]
    cropped_img = img[coord_x:coord_x+width, coord_y:coord_y+height, :]
    return cropped_img, cropped_label



cropped_img_name = []
cropped_label_name = []
for x in data_images:
    img_id, img, mask, _, _ = x
    index = images_id.index(img_id)
    if index%10==0: print(index)
    cropped_img, cropped_label = cropper(x, 256, 256)
    tiff.imsave('images/train_cropped_entropy/'+img_id+'.tif', cropped_img)
    tiff.imsave('images/label_cropped_entropy/'+img_id+'.png', cropped_label)
    cropped_img_name.append(img_id+'.tif')
    cropped_label_name.append(img_id+'.png')  
    

def create_cropped_training(x):
    img_id, img, mask, _, _ = x
    index = images_id.index(img_id)
    print(index)
    cropped_img, cropped_label = cropper(x, 256, 256)
    tiff.imsave('images/images_cropped_entropy/'+img_id+'.tif', cropped_img)
    tiff.imsave('images/label_cropped_entropy/'+img_id+'.png', cropped_label)
    cropped_img_name.append(img_id+'.tif')
    cropped_label_name.append(img_id+'.png')    
    return(cropped_img_name, cropped_label_name)

# parallel version
l = Parallel(n_jobs=5)(delayed(create_cropped_training)(x) for x in data_images)


train_img_name = [os.path.basename(x) for x in glob('images/train_cropped_entropy/*.tif')]
label_img_name = [os.path.basename(x) for x in glob('images/label_cropped_entropy/*.png')]

train_test = train_test_split(train_img_name, label_img_name, test_size=0.4, random_state = 1)

df_train_cropped = pd.DataFrame(data={'im':train_test[0], 'label':train_test[2]})
df_train_cropped.to_csv('images/train_cropped.csv', header=False, index=False)

df_test_cropped = pd.DataFrame(data={'im':train_test[1], 'label':train_test[3]})
df_test_cropped.to_csv('images/test_cropped.csv', header=False, index=False)



# crop images without entropy
for i in range(len(data_images)):
    img_id, img, mask, _, _ = data_images[i]
    print('i={}, img_id = {}'.format(i, img_id))
    height, width = img.shape[:2]
    x_crops, y_crops = width//256, height//256
    #print('x_crops {}, y_crops {}'.format(x_crops, y_crops))
    for x in range(x_crops):
        for y in range(y_crops):
            #print('x={}, y={}'.format(x,y))            
            cropped_img = img[y*256:(y+1)*256, x*256:(x+1)*256, :]
            cropped_label = mask[y*256:(y+1)*256, x*256:(x+1)*256]
            if 1 in cropped_label and cropped_label.shape==(256, 256):
                print('success!')
                tiff.imsave('images/images_cropped256/'+img_id+'_'+str(x)+'_'+str(y)+'.tif',
                            cropped_img)
                tiff.imsave('images/label_cropped256/'+img_id+'_'+str(x)+'_'+str(y)+'.png',
                            cropped_label)

images_img_name = [os.path.basename(x) for x in glob('images/images_cropped256/*.tif')]
label_img_name = [os.path.basename(x) for x in glob('images/label_cropped256/*.png')]

train_test = train_test_split(images_img_name, label_img_name, test_size = 0.1, random_state = 1)

df_train_cropped = pd.DataFrame(data={'im':train_test[0], 'label':train_test[2]})
df_train_cropped.to_csv('images/train_cropped256.csv', header=False, index=False)

df_test_cropped = pd.DataFrame(data={'im':train_test[1], 'label':train_test[3]})
df_test_cropped.to_csv('images/test_cropped256.csv', header=False, index=False)


# test
a = tiff.imread('images/images_cropped256/11ska445785_14_0.tif')
b = tiff.imread('images/label_cropped256/11ska445785_14_0.png')
    
tiff.imshow(a)
tiff.imshow(255*np.stack([b,b,b]))    
    
