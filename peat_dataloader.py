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
    def __init__(self):

        self.images = os.listdir("/home/ajay/Documents/dataset_tif/final_data")
        self.masks = os.listdir("/home/ajay/Documents/dataset_mask/final_data")

        assert len(self.images)==len(self.masks), "lengths are not matching"
        


        self.transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        
        image_path = os.path.join("/home/ajay/Documents/dataset_tif/final_data", self.images[idx])
        mask_path = os.path.join("/home/ajay/Documents/dataset_mask/final_data", self.images[idx])

       
        

        image = Image.open(image_path)

        mask = Image.open(mask_path)

        mask = mask.filter(ImageFilter.MaxFilter(7))

        # import pdb; pdb.set_trace()
        

        return self.transform(image),self.transform(mask)


