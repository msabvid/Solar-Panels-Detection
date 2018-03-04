#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:27:08 2017

@author: msabate
"""

import os
from os.path import join
import rasterio
import numpy as np
from scipy import stats
import fiona
from glob import glob
import re
import sklearn
from collections import Counter
import itertools




PROJECT_DIR = '/Users/msabate/Projects/CityFinancial'
SENTINEL_WORKING_DIR = join(PROJECT_DIR, 'data', 'Sentinel', 'working')
SENTINEL_INPUT_DIR = join(PROJECT_DIR, 'data', 'Sentinel', 'input')
WGS84_DIR = join(PROJECT_DIR, 'data', 'Sentinel', 'working',)
SHAPEFILE_PATH = '/Users/msabate/Projects/CityFinancial/data/OSM/england-latest-free/gis.osm_water_a_free_1.shp'
SHAPEFILE_PATH = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/gis.osm_landuse_a_free_1.shp'

os.getcwd()
os.chdir(PROJECT_DIR)

with rasterio.open('S2A_MSIL1C_20170222T104031_N0204_R008_T31TDF_20170222T104801_resampled_RGB.tif') as src:
    src.width
    src.height
    src.bounds
    src.transform
    src.crs
    src.count
    

    
with rasterio.open('S2A_MSIL1C_20170222T104031_N0204_R008_T31TDF_20170222T104801_resampled_RGB.tif') as src:
   a = src.read()
   print(src.width, src.height)
   print(src.bounds)
   print(src.transform)
   print(src.crs)
   print(src.count)
    


with rasterio.open('data/Sentinel/working/working.tif') as src:
    a = src.read()

    
r, g, b = a[0,:,:], a[1,:,:], a[2,:,:]


img = np.stack([r,g,b])
tiff.imshow(img)




def reproject_dataset(geotiff_path):
    """Project a GeoTIFF to the W
    
    
    S84 coordinate reference system.
    See https://mapbox.github.io/rasterio/topics/reproject.html"""

    # We want to project the GeoTIFF coordinate reference system (crs)
    # to WGS84 (e.g. into the familiar Lat/Lon pairs). WGS84 is analogous
    # to EPSG:4326
    dst_crs = 'EPSG:4326'

    with rasterio.open(geotiff_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        satellite_img_name, _ = os.path.splitext(os.path.basename(geotiff_path))
        out_file_name = "{}_wgs84.tif".format(satellite_img_name)
        out_path = os.path.join(WGS84_DIR, out_file_name)
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest)
    
        return rasterio.open(out_path), out_path



water_features = np.empty((0,))
with fiona.open(SHAPEFILE_PATH) as shapefile:
    geometries = [feature['geometry'] for feature in shapefile]
    water_features = np.concatenate((water_features, geometries), axis=0)
    

geotiff_path = join(SENTINEL_WORKING_DIR, 'S2A_MSIL1C_20170222T104031_N0204_R008_T31TDF_20170222T104801_resampled_RGB.tif')
geotiff_path = join(SENTINEL_WORKING_DIR, 'working.tif')

raster_dataset, wgs84_path = reproject_dataset(geotiff_path)

bitmap_image = rasterio.features.rasterize(
    ((g, 255) for g in water_features),
    out_shape=raster_dataset.shape,
    transform=raster_dataset.transform)

import tifffile as tiff
tiff.imsave(join(SENTINEL_WORKING_DIR, 'mask.tif'), bitmap_image)


'Different features in the shape files'
SHAPEFILE_ENGLAND = '/Users/msabate/Projects/CityFinancial/data/OSM/england-latest-free/'
l_shp = glob(SHAPEFILE_ENGLAND + '*.shp')

list_classes = []

for shapefile in l_shp:
    with fiona.open(shapefile) as src:
        for p in src:
            list_classes.append(p['properties']['fclass'])
            
set_classes = set(list_classes)

r = re.compile(".*forest.*")
newlist = filter(r.match, a)



# Features in land-use shape files from Scotland
SHAPEFILE_SCOTLAND = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/'
l_shp = glob(SHAPEFILE_SCOTLAND + '*.shp')

land_use = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/gis.osm_landuse_a_free_1.shp'

list_classes = []
src = fiona.open(land_use)
for p in src:
    list_classes.append(p['properties']['fclass'])

src.close()

counter = Counter(list_classes)
counter.most_common()


# we create land-use features for scotland
mask_feature = 'residential'
SHAPEFILE_PATH = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/gis.osm_landuse_a_free_1.shp'
features = np.empty((0,))
with fiona.open(SHAPEFILE_PATH) as shapefile:
    geometries = [feature['geometry'] for feature in shapefile if feature['properties']['fclass']==mask_feature]
    features = np.concatenate([features, geometries], axis=0)

geotiff_path = join(SENTINEL_WORKING_DIR, 'edinburgh.tif')
raster_dataset, wgs84_path = reproject_dataset(geotiff_path)

bitmap_image = rasterio.features.rasterize(
    ((g, 255) for g in features),
    out_shape=raster_dataset.shape,
    transform=raster_dataset.transform)


tiff.imsave(join(SENTINEL_WORKING_DIR, 
                 os.path.splitext(os.path.basename(geotiff_path))[0] + '_mask_'+mask_feature+'.tif'), bitmap_image)


# FEATURES in roads
roads_shp = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/gis.osm_roads_free_1.shp'

list_classes = []
src = fiona.open(roads_shp)
for p in src:
    list_classes.append(p['properties']['fclass'])

src.close()

counter = Counter(list_classes)
counter.most_common()

# we create roads features for scoltands
SHAPEFILE_PATH = '/Users/msabate/Projects/CityFinancial/data/OSM/scotland-latest-free/gis.osm_roads_free_1.shp'
features = np.empty((0,))
with fiona.open(SHAPEFILE_PATH) as shapefile:
    geometries = [feature['geometry'] for feature in shapefile]
    features = np.concatenate([features, geometries], axis=0)

geotiff_path = join(SENTINEL_WORKING_DIR, 'edinburgh.tif')
raster_dataset, wgs84_path = reproject_dataset(geotiff_path)

bitmap_image = rasterio.features.rasterize(
    ((g, 255) for g in features),
    out_shape=raster_dataset.shape,
    transform=raster_dataset.transform)


tiff.imsave(join(SENTINEL_WORKING_DIR, 
                 os.path.splitext(os.path.basename(geotiff_path))[0] + '_mask_roads.tif'), bitmap_image)





