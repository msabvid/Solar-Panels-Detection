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




