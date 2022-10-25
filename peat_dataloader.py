import os
import torch
import shutil
import numpy as np

from PIL import ImageFilter

from PIL import Image
import cv2


from tqdm import tqdm
from urllib.request import urlretrieve

from torchvision import transforms


# from skimage import io



class PeatDataset(torch.utils.data.Dataset):
    def __init__(self, data_split, transform = None):

        if data_split ==  'train' :

            self.images = os.listdir("/home/ajay/Documents/Copyof_data/dataset_tif/split_dataset/train/dataset")
            self.masks = os.listdir("/home/ajay/Documents/Copyof_data/dataset_mask/split_dataset/train/dataset")

        if data_split ==  'val' :

            self.images = os.listdir("/home/ajay/Documents/Copyof_data/dataset_tif/split_dataset/val/dataset")
            self.masks = os.listdir("/home/ajay/Documents/Copyof_data/dataset_mask/split_dataset/val/dataset")

        if data_split ==  'test' :

            self.images = os.listdir("/home/ajay/Documents/Copyof_data/dataset_tif/split_dataset/test/dataset")
            self.masks = os.listdir("/home/ajay/Documents/Copyof_data/dataset_mask/split_dataset/test/dataset")

        assert len(self.images)==len(self.masks), "lengths are not matching"
        

        # mean = (0.5, 0.5, 0.5)
        # std = (0.5, 0.5, 0.5)

        self.transform = transform

        # self.transform=transforms.Compose([
        #     # transforms.normalize(mean, std),
        #     transforms.Resize(256),
        #     transforms.ToTensor()
        # ])

        # self.transform = A.Compose([
        #     A.Resize(256,256), 
        #     albumentations.pytorch.transforms.ToTensorV2 ()


        # ])

        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        
        image_path = os.path.join("/home/ajay/Documents/Copyof_data/dataset_tif/dataset_256/dataset", self.images[idx])
        mask_path = os.path.join("/home/ajay/Documents/Copyof_data/dataset_mask/dataset_256/dataset", self.images[idx])

        # import pdb; pdb.set_trace()

        # For PIL
        image = Image.open(image_path)
        # image = np.array(image)

        mask = Image.open(mask_path)
        mask = mask.filter(ImageFilter.MaxFilter(7))
        # mask = np.array(mask)
        # mask = mask[:, :, np.newaxis]

        # For cv2
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)//255
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        
        # print(image.dtype, mask.dtype)

        # import pdb; pdb.set_trace()

        # if self.transform is not None:
        #     transformed = self.transform(image=image, mask=mask)
        #     image = transformed["image"]
        #     mask = transformed["mask"]

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

   


        return image, mask


