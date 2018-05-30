# Solar-Panels-Detection
* Detection of Solar Panels from high-resolution aerial images in https://figshare.com/articles/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780. We train [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) implemented with PyTorch.
  * Benchmark study of U-Net training using [Hogwild](https://arxiv.org/abs/1106.5730) and MPI
* Creation of training set for other detection problems using Sentinel-2 images and Open Street Maps

## Scripts
* _src/data_loader.py_: classes to load 256x256 images in the training set
* _src/utils/solar_panels_detection_california.py_: creation of training set using geojson file and aerial images from [here](https://figshare.com/articles/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780).
* _src/train_unet2.py_: training of U-Net using Cuda Tensors
* _src/train_unet2_cpu.py_: training of U-Net using cpu Tensors
* _src/Hogwild/train_unet2_cpu_Hogwild.py_: distributed training of U-Net in one node of a cluster, [doing asynchrnous update of model parameters](https://arxiv.org/abs/1106.5730).
* _src/mpi/train_unet2_cpu_mpi.py_: distributed training of U-Net in several nodes of a cluster using mpi4py
* _src/OpenStreetMaps/osm.py_: rasterisaton of OpenStreetMaps data to create mask images of Sentinel-2 images. Useful to  create training sets for other detection problems 

## Data
* Solar panels locations in aerial images of four cities in California: https://figshare.com/articles/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780
* Sentinel-2 images https://scihub.copernicus.eu
* OpenStreetMaps: shapefiles of different geographical features from Scotland: http://download.geofabrik.de/europe/great-britain.html

## Results
### True Positives examples
- Left: Original image. Center: U-Net output. Right: Solar panels delimited using U-Net output
![TP1](/images/TP1.png)
![TP2](/images/TP2.png)
![TP3](/images/TP3.png)
![TP4](/images/TP4.png)

### False Positives Examples
![FP1](/images/FP1.png)
![FP2](/images/FP2.png)

### Sentinel-2 land monitoring training set creation
- Sentinel-2 image of Edinburgh area
![Sentinel2](/images/Sentinel_edi.png)
- Mask image with white pixels at forests location
![forest](/images/edi_forest.png)
- Mask image with white pixels at roads location
![roads](/images/edi_roads.png)
