import os
import torch
import time
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from peat_dataloader import PeatDataset

from torchvision.utils import save_image


train_dataset = PeatDataset()


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)


device = torch.device("cuda:0") 

model = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=1, activation=None, aux_params=None)
model=model.to(device)

# criterion = smp.losses.DiceLoss(mode='binary', classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)
criterion = smp.losses.DiceLoss(mode='binary', log_loss=True)

criterion=criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float('inf')

model_version = '0.00'


# for epoch in range(10):
#     for image,mask in train_dataloader:
#         image=image.to(device)
#         mask=mask.to(device)
#         pred = model(image) 

#         loss = criterion(pred, mask)
#         print(loss.item())

#         optimizer.zero_grad()

#         loss.backward()
#         optimizer.step()


#     # print images


def train_model(model, train_dataloader, optimizer, epoch):
    _ = model.train()
    loss_total = 0
    total = 0 
    if torch.cuda.is_available():
        model.cuda()
    

    for image,mask in train_dataloader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()
            

        prediction = model.forward(image)

        # print(prediction.shape)
        # import pdb; pdb.set_trace() 
        
        
        loss = criterion(prediction, mask)

        loss_total += loss/16

        print(loss.item())

        total += 1

        loss.backward()
        optimizer.step()
        
    save_image(prediction, f'outputs/prediction{epoch}.png')
    save_image(image, f'outputs/image{epoch}.png')
    save_image(mask, f'outputs/mask{epoch}.png')
    
    return loss_total/total


for epoch in range(50):
    loss = train_model(model, train_dataloader, optimizer, epoch )

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
