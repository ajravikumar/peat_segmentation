import os
import io

import torch
from peat_dataloader import PeatDataset
from torch.utils.data import DataLoader


from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pdb

device = torch.device("cuda:0")

mrnet = torch.load(f'../models/model_0.00_0.49_15_03.pth')
mrnet = mrnet.to(device)

_ = mrnet.eval()

train_dataset = PeatDataset()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)






