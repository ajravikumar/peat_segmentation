import os
import torch
import shutil
import numpy as np

from PIL import ImageFilter

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

from torchvision import transforms


# from skimage import io



class PeatDataset(torch.utils.data.Dataset):
    def __init__(self, data_split):

        if data_split ==  'train' :

            self.images = os.listdir("/home/ajay/Documents/dataset_tif/split_dataset/train/dataset")
            self.masks = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/train/dataset")

        if data_split ==  'val' :

            self.images = os.listdir("/home/ajay/Documents/dataset_tif/split_dataset/val/dataset")
            self.masks = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/val/dataset")

        if data_split ==  'test' :

            self.images = os.listdir("/home/ajay/Documents/dataset_tif/split_dataset/test/dataset")
            self.masks = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/test/dataset")

        assert len(self.images)==len(self.masks), "lengths are not matching"
        

        # mean = (0.5, 0.5, 0.5)
        # std = (0.5, 0.5, 0.5)
        self.transform=transforms.Compose([
            # transforms.normalize(mean, std),
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        
        image_path = os.path.join("/home/ajay/Documents/dataset_tif/dataset_256/dataset", self.images[idx])
        mask_path = os.path.join("/home/ajay/Documents/dataset_mask/dataset_256/dataset", self.images[idx])

        # import pdb; pdb.set_trace()

        # if self.data_split ==  'train' :

        #     image_path = os.path.join("/home/ajay/Documents/dataset_tif/split_dataset/train/dataset", self.images[idx])
        #     mask_path = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/train/dataset", self.images[idx])

        # if self.data_split ==  'val' :

        #     image_path = os.path.join("/home/ajay/Documents/dataset_tif/split_dataset/val/dataset", self.images[idx])
        #     mask_path = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/val/dataset", self.images[idx])

        # if self.data_split ==  'test' :

        #     image_path = os.path.join("/home/ajay/Documents/dataset_tif/split_dataset/test/dataset", self.images[idx])
        #     mask_path = os.listdir("/home/ajay/Documents/dataset_mask/split_dataset/test/dataset", self.images[idx])
        

        image = Image.open(image_path)

        mask = Image.open(mask_path)

        mask = mask.filter(ImageFilter.MaxFilter(7))

        # import pdb; pdb.set_trace()
        

        return self.transform(image),self.transform(mask)


