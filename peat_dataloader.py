import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

from torchvision import transforms


# from skimage import io



class PeatDataset(torch.utils.data.Dataset):
    def __init__(self):

        # assert mode in {"train", "valid", "test"}

        # self.root = root
        # self.mode = mode
        # self.transform = transform



        self.images = os.listdir("/home/ajay/Documents/dataset_tif/Data/Raheenmore")
        self.masks = os.listdir("/home/ajay/Documents/dataset_mask/final_data")

        assert len(self.images)==len(self.masks), "lengths are not matching"
        


        self.transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        # self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        
        image_path = os.path.join("/home/ajay/Documents/dataset_tif/Data/Raheenmore", self.images[idx])
        mask_path = os.path.join("/home/ajay/Documents/dataset_mask/final_data", self.images[idx])

        # image = io.imread(image_path)
        # mask = io.imread(mask_path)
        

        image = Image.open(image_path)

        mask = Image.open(mask_path)
        # mask = self._preprocess_mask(trimap)

        # sample = dict(image=image, mask=mask, trimap=trimap)
        # if self.transform is not None:
        #     sample = self.transform(**sample)

        return self.transform(image),self.transform(mask)


#     @staticmethod
#     def _preprocess_mask(mask):
#         mask = mask.astype(np.float32)
#         mask[mask == 2.0] = 0.0
#         mask[(mask == 1.0) | (mask == 3.0)] = 1.0
#         return mask

#     def _read_split(self):
#         split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
#         split_filepath = os.path.join(self.root, "annotations", split_filename)
#         with open(split_filepath) as f:
#             split_data = f.read().strip("\n").split("\n")
#         filenames = [x.split(" ")[0] for x in split_data]
#         # import pdb; pdb.set_trace()
#         if self.mode == "train":  # 90% for train
#             filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
#         elif self.mode == "valid":  # 10% for validation
#             filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
#         return filenames

#     @staticmethod
#     def download(root):

#         # load images
#         filepath = os.path.join(root, "images.tar.gz")
#         download_url(
#             url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
#             filepath=filepath,
#         )
#         extract_archive(filepath)

#         # load annotations
#         filepath = os.path.join(root, "annotations.tar.gz")
#         download_url(
#             url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
#             filepath=filepath,
#         )
#         extract_archive(filepath)


# class SimpleOxfordPetDataset(OxfordPetDataset):
#     def __getitem__(self, *args, **kwargs):

#         sample = super().__getitem__(*args, **kwargs)

#         # resize images
#         image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
#         mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
#         trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

#         # convert to other format HWC -> CHW
#         sample["image"] = np.moveaxis(image, -1, 0)
#         sample["mask"] = np.expand_dims(mask, 0)
#         sample["trimap"] = np.expand_dims(trimap, 0)

#         return sample


# class TqdmUpTo(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, filepath):
#     directory = os.path.dirname(os.path.abspath(filepath))
#     os.makedirs(directory, exist_ok=True)
#     if os.path.exists(filepath):
#         return

#     with TqdmUpTo(
#         unit="B",
#         unit_scale=True,
#         unit_divisor=1024,
#         miniters=1,
#         desc=os.path.basename(filepath),
#     ) as t:
#         urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
#         t.total = t.n


# def extract_archive(filepath):
#     extract_dir = os.path.dirname(os.path.abspath(filepath))
#     dst_dir = os.path.splitext(filepath)[0]
#     if not os.path.exists(dst_dir):
#         shutil.unpack_archive(filepath, extract_dir)
