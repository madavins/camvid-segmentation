import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os
import cv2
import albumentations as A


class Dataset(BaseDataset):
    """CamVid Dataset. 
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): index of the classes that we want to extract
    """
            
    def __init__(self, images_dir, masks_dir, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.ids]
        self.augmentation = augmentation
        
    def __getitem__(self, i):   
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        image = image.transpose((2, 0, 1))
        if image.max() > 1:
            image = image / 255
        
        #RETURN MODE TENSOR O MODE NUMPY
        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)
        
        #return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
