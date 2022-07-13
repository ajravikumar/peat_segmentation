import os
import torch
import time
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np

from sklearn.model_selection import train_test_split

from pprint import pprint
from torch.utils.data import DataLoader

from peat_dataloader import PeatDataset

from torchvision.utils import save_image

import neptune.new as neptune


run = neptune.init(project='ajravikumar/peat-segmentation')

params = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "criterion": "DiceLoss",
    "resolution": "256x256"
}
run["parameters"] = params



train_dataset = PeatDataset()

# import pdb; pdb.set_trace()

train_data, val_data = train_test_split(train_dataset, test_size=0.25)

print(len(train_dataset))
print(len(train_data))
print(len(val_data))

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=16, num_workers=8, drop_last=True)

device = torch.device("cuda:0") 

model = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=1, activation=None, aux_params=None)
model=model.to(device)

# criterion = smp.losses.DiceLoss(mode='binary', classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)
criterion = smp.losses.DiceLoss(mode='binary', log_loss=True)

criterion=criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float('inf')

model_version = '0.00'



def train_model(model, train_dataloader, optimizer, epoch):
    _ = model.train()
    loss_total = 0
    total = 0 
    if torch.cuda.is_available():
        model.cuda()
    
    losses = []

    for image,mask in train_dataloader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()
            

        prediction = model.forward(image)

        # print(prediction.shape)
        # import pdb; pdb.set_trace() 
        
        
        loss = criterion(prediction, mask)

        # loss_total += loss/16
        
        print(loss.item())

        loss_value = loss.item()
        losses.append(loss_value)

        

        loss.backward()
        optimizer.step()
        
    save_image(prediction, f'outputs/prediction{epoch}.png')
    save_image(image, f'outputs/image{epoch}.png')
    save_image(mask, f'outputs/mask{epoch}.png')

    train_loss_epoch = np.round(np.mean(losses), 4)
    
    return train_loss_epoch


def val_model (model, val_dataloader, optimizer, epoch):

    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()
    
    losses = []

    for image,mask in val_dataloader:
        
        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()
            

        prediction = model.forward(image)
        
        
        loss = criterion(prediction, mask)

        # loss_total += loss/16
        
        # print(loss.item())

        loss_value = loss.item()
        losses.append(loss_value)

        

        loss.backward()
        optimizer.step()
    
    val_loss_epoch = np.round(np.mean(losses), 4)
    
    return val_loss_epoch



for epoch in range(50):
    loss = train_model(model, train_dataloader, optimizer, epoch )

    val_loss = val_model(model, val_dataloader, optimizer, epoch)

    run["train/loss"].log(loss)
    run["validation/loss"].log(val_loss)
    # run.stop()

    print('total loss is' , loss.item())

    # if loss.item() < best_loss and loss.item() != 0.0:
    #     best_loss = loss.item()
    #     model_name = "{}_{:.2f}_{}".format(model_version,loss.item(), time.strftime("%d_%m"))
    #     file_name = f'model_{model_name}.pth'

    #     for f in os.listdir('./models/'):
    #             # print(f)
    #             if model_version in f:
    #                 os.remove(f'./models/{f}')
        
    #     torch.save(model, f'./models/{file_name}')
