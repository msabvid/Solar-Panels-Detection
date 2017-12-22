# Solar-Panels-Detection
* Detection of Solar Panels from high-resolution aerial images in https://figshare.com/articles/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780. We train [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) implemented with PyTorch.
  * Benchmark study of U-Net training using [Hogwild](https://arxiv.org/abs/1106.5730) and MPI
* Creation of training set for other detection problems using Sentinel-2 images and Open Street Maps

## Scripts
* _src/data_loader.py_: classes to load 256x256 images in the training set
* _src/solar_panels_detection_california.py_: creation of training set using geojson file and aerial images from [here](https://figshare.com/articles/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780).
* _src/train_unet2.py_: training of U-Net
* _src/train_unet2_cpu_Hogwild.py_: distributed training of U-Net in one node of a cluster, doing asynchrnous update of model parameters
* _src/train_unet2_cpu_mpi.py_: distributed training of U-Net in several nodes of a cluster using mpi4py
