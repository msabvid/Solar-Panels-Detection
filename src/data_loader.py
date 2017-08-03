import torch.utils.data as data

from PIL import Image
import os
import os.path
#from tifffile import imread
from scipy.misc import imread, imresize
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
import tifffile as tiff

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, dir_output,text_file):
    images = []
    print(text_file)
    with open(text_file, 'r') as f:
        for line in f:
            line = line[:-1]
            #print(line)
            split_lines = line.split(",")
            path1 = '{0}{1}'.format(dir, split_lines[0])
            path2 = '{0}{1}'.format(dir_output, split_lines[1])
            item = (path1,path2)
            images.append(item)
    random.shuffle(images)
    if len(images)>2000:
        return images[:2000]
    else:
        return images[:900]


def make_dataset_from_big_image(dir_subimages, filename):
    """
    This function does the following:
        1) crops big images into subimages of 256x256,
        2) saves subimages in dir_subimages
        3) creates the images_csv list with the path of the subimages
    
    Input:
    -----
        dir_subimages: path to save images
        filename: path+filename of the big image
    
    Output:
    ------
        images_csv: list of the subimages paths
        
    """

    image = imread(filename)
    img_id = os.path.splitext(os.path.basename(filename))[0]
    height, width = image.shape[:2]
    n_rows, n_cols = height//256, width//256
    images_csv= []
    for i in range(n_rows):
        for j in range(n_cols):
            cropped = image[i*256:(i*256)+256, j*256:(j*256)+256, :]
            cropped_id = img_id + '_'+str(i)+'_'+str(j) + '.tif' 
            # save image
            tiff.imsave(os.path.join(dir_subimages, cropped_id), cropped)
            # write in csv file image path
            images_csv.append(os.path.join(dir_subimages, cropped_id))
            
        if n_cols*256 < width:
            cropped = image[i*256:(i*256)+256, width-256:width,:]
            cropped_id = img_id + '_'+str(i)+'_'+str(j+1)+'.tif'
            # save image
            tiff.imsave(os.path.join(dir_subimages, cropped_id), cropped)
            # write in csv file image path
            images_csv.append(os.path.join(dir_subimages, cropped_id))
            
            
    if n_rows*256 < height:
        for j in range(n_cols):
            cropped = image[height-256:height, j*256:(j*256)+256, :]
            cropped_id = img_id + '_'+str(i+1)+'_'+str(j)+'.tif' 
            # save image
            tiff.imsave(os.path.join(dir_subimages, cropped_id), cropped)
            # write in csv file image path
            images_csv.append(os.path.join(dir_subimages, cropped_id))

            
        if n_cols*256 < width:
            cropped = image[height-256:height, width-256:width,:]
            cropped_id = img_id + '_'+str(i+1)+'_'+str(j+1)+'.tif'
            # save image
            tiff.imsave(os.path.join(dir_subimages, cropped_id), cropped)
            # write in csv file image path
            images_csv.append(os.path.join(dir_subimages, cropped_id))
    
    if len(images_csv)%5 > 0:
        for i in range(5-(len(images_csv)%5)):
            images_csv.append(images_csv[-1])
            
    return images_csv




def reconstruct_image(dir_subimages, filename):
    """
    Input:
    -----
        dir_subimages: path of the subimages
        filename: path+filename of the big original image
    
    Output:
    ------
        reconstructed image
        
    """
    original_image = tiff.imread(filename)
    height, width = original_image.shape[:2]
    n_rows, n_cols = height//256, width//256
    
    #subimages = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(dir_subimages, '*.png'))]
    img_id = os.path.splitext(os.path.basename(filename))[0]
    img_rows = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            img = tiff.imread(os.path.join(dir_subimages, img_id+'_'+str(i)+'_'+str(j)+'.png'))
            if (j == 0):
                img_row = img
            else:
                img_row = np.concatenate([img_row, img], axis = 2)
        if n_cols*256 < width:
            img = tiff.imread(os.path.join(dir_subimages, img_id+'_'+str(i)+'_'+str(j+1)+'.png'))
            img = img[:,:,-width%256:]
            img_row = np.concatenate([img_row, img], axis = 2)
        img_rows.append(img_row)
    if n_rows*256 < height:
        for j in range(n_cols):
            img = tiff.imread(os.path.join(dir_subimages, img_id+'_'+str(i+1)+'_'+str(j)+'.png'))
            img = img[:,-height%256:,:]
            if (j == 0):
                img_row = img
            else:
                img_row = np.concatenate([img_row, img], axis = 2)
        if n_cols*256 < width:
            img = tiff.imread(os.path.join(dir_subimages, img_id+'_'+str(i+1)+'_'+str(j+1)+'.png'))
            img = img[:,-height%256:,-width%256:]
            img_row = np.concatenate([img_row, img], axis = 2)            
        img_rows.append(img_row)
    
    reconstruction = np.concatenate(img_rows, axis=1)
    return reconstruction





def default_loader(path):
    return imread(path)
    #return(Image.open(path))

class ImagerLoader(data.Dataset):
    def __init__(self, root, root_output,text_file,transform=None, target_transform=None,
                 loader=default_loader, crop=False, normalize = False, size_cropped = 512):

        imgs = make_dataset(root, root_output,text_file)

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = transform
        self.loader = loader
        self.crop = crop
        self.normalize = normalize
        self.size_cropped = size_cropped

    def __getitem__(self, index):
        path, path_output = self.imgs[index]
        img = self.loader(path)#.astype(int)   image has dimension height x width x n_channels
        output = self.loader(path_output)#.astype(int)
        #img = imresize(img, (512, 512))
        #output = imresize(output, (512, 512))
        
        img = img.astype('int16')
        output = output.astype('int16')
        
        # if we want to crop the image at the centre 
        if self.crop:
            h,w,channels = img.shape
            img = img[(h//2-self.size_cropped//2):(h//2+self.size_cropped//2), (w//2-self.size_cropped//2):(w//2+self.size_cropped//2),:]   
            
            h,w = output.shape
            output = output[h//2-self.size_cropped//2:h//2+self.size_cropped//2, w//2-self.size_cropped//2:w//2+self.size_cropped//2]
        
        img = np.transpose(img, (2,0,1))
        
        # if we want to normalize the images to [-1,1]
        if self.normalize:
            img = img.astype(float)
            img = (img-128)/128
            img = torch.FloatTensor(img)
        else: 
            img = torch.ShortTensor(img)
            
#        if self.transform is not None:
#           img = self.transform(img)
#        if self.target_transform is not None:
#            output = self.target_transform(output)
        
        img = img.float()
        img_id = os.path.basename(path).split('.')[0]
        return img_id, img, torch.ShortTensor(output).long()

    def __len__(self):
        return len(self.imgs)


class ImageLoaderPredictionBigImage(data.Dataset):
    def __init__(self, dir_subimages, filename, normalize = False, loader=default_loader):

        imgs = make_dataset_from_big_image(dir_subimages, filename)

        self.imgs = imgs
        self.dir_subimages = dir_subimages
        self.filename = filename
        self.loader = loader
        self.normalize = normalize
    
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        img = img.astype('int16')
        img = np.transpose(img, (2,0,1))
        
        if self.normalize:
            img = img.astype(float)
            img = (img-128)/128.
        
        img = torch.FloatTensor(img)
        img_id = os.path.basename(path).split('.')[0]
        return img_id, img

    def __len__(self):
        return len(self.imgs)


