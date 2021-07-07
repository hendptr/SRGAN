import torch 
import torch.nn as nn 
import torch.optim 
from torch.utils.data import DataLoader 

from math import log10

from models import SRCNN 
from dataset import ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)

#PARAMETER
batch_size = 4
epochs = 100
lr = 1e-4

train_dataset = ImageDataset('T91', 2, 96)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([
    {"params": model.feature.parameters(),
     "params": model.remap.parameters(),
     "params": model.reconstruction.parameters(), "lr":1e-5
    }
], lr=lr)


def train():
    for epoch in range(epochs):
        cur_loss = 0
        psnr_loss = 0
        for idx, (lr_img, or_img) in enumerate(train_loader):
            lr_img = lr_img.to(device)
            or_img = or_img.to(device)

            optimizer.zero_grad()
            out = model(lr_img)
            loss = criterion(out, or_img)

            cur_loss+=loss.item()
            psnr_loss+=10*log10(1/loss.item())

            loss.backward()
            optimizer.step()

        print('[EPOCH {}] -- Loss {:.5f} -- PSNR {:.5f} dB'.format(epoch+1, cur_loss/len(train_loader), psnr_loss/len(train_loader)))


def save_checkpoint(model, optimizer):
    params = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
        }

    torch.save(params, 'model.pth')




train()
save_checkpoint(model, optimizer)