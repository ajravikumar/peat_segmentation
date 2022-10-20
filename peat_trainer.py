import os
# from random import sample
import torch
import time
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np

import wandb

from torchvision import transforms
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

from pprint import pprint
from torch.utils.data import DataLoader

from peat_dataloader import PeatDataset

from torchvision.utils import save_image

import neptune.new as neptune


# run = neptune.init(project='ajravikumar/peat-segmentation')

# wandb.init(project='img_seg')



# wandb.config= {
#     "batch_size": 64,
#     "learning_rate": 1e-3,
#     "optimizer": "Adam",
#     "criterion": "DiceLoss",
#     "resolution": "256x256"
# }


train_transform=transforms.Compose([
            # transforms.normalize(mean, std),
            transforms.Resize(256),
            transforms.ToTensor()
        ])

# train_transform=A.Compose([
#             # transforms.normalize(mean, std),
#             A.Resize(256, 256),
#             # ToTensorV2(),
#         ])

val_transform=transforms.Compose([
            # transforms.normalize(mean, std),
            transforms.Resize(256),
            # transforms.ToTensor()
        ])


train_dataset = PeatDataset('train', transform = train_transform)
val_dataset = PeatDataset('val', transform = train_transform)
test_dataset = PeatDataset('test')

# import pdb; pdb.set_trace()

# Assert datasets dont intersect with each other
assert set(test_dataset.images).isdisjoint(set(train_dataset.images))
assert set(test_dataset.images).isdisjoint(set(val_dataset.images))
assert set(train_dataset.images).isdisjoint(set(val_dataset.images))

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
print(f'Number of test samples: {len(test_dataset)}')

num_worker =8

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_worker, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=num_worker, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=num_worker, drop_last=True)



# import pdb; pdb.set_trace()

device = torch.device("cuda:0") 

model = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=1, activation=None, aux_params=None)
model=model.to(device)

# criterion = smp.losses.DiceLoss(mode='binary', classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)
criterion = smp.losses.DiceLoss(mode='binary', log_loss=True)

criterion=criterion.to(device)

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold= 1e-3 )
best_loss = float('inf')

model_version = '0.00'



def train_model(model, train_dataloader, optimizer, epoch):
    _ = model.train()
     
    if torch.cuda.is_available():
        model.cuda()
    
    losses = []
    iou = []

    for image,mask in train_dataloader:
        optimizer.zero_grad()

        # image = image.float()

        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()
            
        

        prediction = model.forward(image)

        # import pdb; pdb.set_trace()
        
        loss = criterion(prediction, mask)

        
        out = (prediction > 0.5).int()
        mask1 = (mask > 0.5).int()
        # import pdb; pdb.set_trace()
        tp, fp, fn, tn = smp.metrics.functional.get_stats(out, mask1, mode='binary', threshold = None)
        iou_score = smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction='micro')

        # print(f'IOU score is {iou_score}')

        # print(loss.item())

        loss_value = loss.item()
        losses.append(loss_value)

        iou_value = iou_score.item()
        iou.append(iou_value)
        

        loss.backward()
        optimizer.step()
        
    # save_image(prediction, f'outputs/prediction{epoch}.png')
    # save_image(image, f'outputs/image{epoch}.png')
    # save_image(mask, f'outputs/mask{epoch}.png')

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_iou = np.round(np.mean(iou) ,4)
    # print('''[Epoch: {0} / {1} | avg train loss {2} | lr : {3}'''.
    #               format(
    #                   epoch + 1,
    #                   30,
    #                   np.round(np.mean(losses), 4),
    #                   1e-3
    #               ))
    
    return train_loss_epoch, train_iou


def val_model (model, val_dataloader, epoch):

    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()
    
    losses = []
    iou = []

    with torch.no_grad():
        for image,mask in val_dataloader:
            
            if torch.cuda.is_available():
                image = image.cuda()
                mask = mask.cuda()
                

            prediction = model.forward(image)
            
            
            loss = criterion(prediction, mask)

            # loss_total += loss/16
            
            # print(loss.item())

            out = (prediction > 0.5).int()
            mask1 = (mask > 0.5).int()
            
            tp, fp, fn, tn = smp.metrics.functional.get_stats(out, mask1, mode='binary', threshold = None)
            iou_score = smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction='micro')

            loss_value = loss.item()
            losses.append(loss_value)

            iou_value = iou_score.item()
            iou.append(iou_value)

        

        # loss.backward()
        # optimizer.step()
    
    save_image(prediction, f'outputs/prediction{epoch}.png')
    save_image(image, f'outputs/image{epoch}.png')
    save_image(mask, f'outputs/mask{epoch}.png')


    val_loss_epoch = np.round(np.mean(losses), 4)
    val_iou = np.round(np.mean(iou) ,4)
    
    return val_loss_epoch, val_iou


num_epoch = 50

for epoch in range(num_epoch):

    # print (f'MODEL TRAINING EPOCH:{epoch}')
    loss, iou = train_model(model, train_dataloader, optimizer, epoch )

    val_loss, val_iou = val_model(model, val_dataloader, epoch)

    # run["train/loss"].log(loss)
    # run["validation/loss"].log(val_loss)
    # run.stop()

    # print('total loss is' , loss.item())

    print('''[Epoch: {0} / {1} | avg train loss {2}  | avg train iou {3} | avg validation loss {4} | avg validation iou {5}| lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epoch,
                      loss,
                      iou,
                      val_loss,
                      val_iou,
                      optimizer.param_groups[0]['lr']
                  ))

    scheduler.step(val_loss)  

    # wandb.log ({"training loss" : loss})  
    # wandb.log ({"training iou" : iou})            
    # wandb.log ({"validation loss" : val_loss})            
    # wandb.log ({"validation iou" : val_iou})                  
          

    # if loss.item() < best_loss and loss.item() != 0.0:
    #     best_loss = loss.item()
    #     model_name = "{}_{:.2f}_{}".format(model_version,loss.item(), time.strftime("%d_%m"))
    #     file_name = f'model_{model_name}.pth'

    #     for f in os.listdir('./models/'):
    #             # print(f)
    #             if model_version in f:
    #                 os.remove(f'./models/{f}')
        
    #     torch.save(model, f'./models/{file_name}')
