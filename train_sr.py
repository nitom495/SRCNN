import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch import device
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from PIL import Image
from pathlib import Path
from collections.abc import Callable
from torch.utils.data import Dataset, DataLoader



class PairedSRTransform:
    def __init__(self,lr_patch_size=128,scale=2):
        self.lr_patch_size=lr_patch_size
        self.scale=scale
    def __call__(self,lr_image,hr_image):
        scale=self.scale

        top,left,height,width=RandomCrop.get_params(lr_image,output_size=(self.lr_patch_size,self.lr_patch_size))
        lr_image=TF.crop(lr_image,top,left,height,width)
        hr_image=TF.crop(hr_image,top*scale,left*scale,height*scale,width*scale)

        lr=TF.to_tensor(lr_image)
        hr=TF.to_tensor(hr_image)

        #Randomly Process to improve accuracy
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])

        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])

        rotations = random.randint(0, 3)
        lr = torch.rot90(lr, rotations, dims=[1, 2])
        hr = torch.rot90(hr, rotations, dims=[1, 2])

        return lr,hr

class PairedSRDataset(Dataset):
    def __init__(self,lr_dir:str,hr_dir:str,scale:int=2,transform:Callable | None=None):
        self.lr_dir=Path(lr_dir)
        self.hr_dir=Path(hr_dir)

        self.scale=scale
        self.transform=transform

        self.lr_paths = sorted(
            self.lr_dir.glob(f"*x{scale}.png")
        )

        self.samples = []

        for lr_path in self.lr_paths:
            image_id = lr_path.stem.removesuffix(f"x{scale}")
            hr_path = self.hr_dir / f"{image_id}.png"

            self.samples.append((lr_path, hr_path))

    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.samples[index]

        lr_image = PIL.Image.open(lr_path).convert("RGB")
        hr_image = PIL.Image.open(hr_path).convert("RGB")

        lr, hr = self.transform(lr_image,hr_image)
        return lr,hr

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,8,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,12,kernel_size=3,padding=1),
            nn.PixelShuffle(2),
        )
    def forward(self,x):
        return self.model(x)

def main():
    device=torch.device("cuda:0")

    train_transform = PairedSRTransform(lr_patch_size=128,scale=2)

    train_dataset = PairedSRDataset(
        lr_dir=(
            r"E:\ScienceLane\SR\dataset"
            r"\DIV2K_train_LR_bicubic\X2"
        ),
        hr_dir=(
            r"E:\ScienceLane\SR\dataset"
            r"\DIV2K_train_HR"
        ),
        scale=2,
        transform=train_transform,
    )

    #lr, hr = train_dataset[0]
    #print(lr.shape)
    #print(hr.shape)
    #print(lr.min(), lr.max())
    #print(hr.min(), hr.max())

    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,drop_last=True,num_workers=4)
    model=NN().to(device)
    #print("Model Device:", next(model.parameters()).device)
    # x=torch.rand(1,3,128,128)
    # y=model(x)
    # print("输入：", x.shape)
    # print("输出：", y.shape)

    loss_fn=nn.L1Loss()

    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.999))

    num_epochs=1

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-6)

    print("start training")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss=0
        for lr_batch,hr_batch in train_loader:
            lr_batch=lr_batch.to(device,non_blocking=True)
            hr_batch=hr_batch.to(device,non_blocking=True)
            # print("LR device:", lr_batch.device)
            # print("HR device:：", hr_batch.device)

            sr_batch=model(lr_batch)
            #print("SR device:", sr_batch.device)

            optimizer.zero_grad()

            loss=loss_fn(sr_batch,hr_batch)
            total_loss=total_loss+loss.item()

            loss.backward()

            optimizer.step()

        scheduler.step()
        average_loss=total_loss/len(train_loader)
        current_lr=scheduler.get_last_lr()[0]

        print(f"Epoch: {epoch+1}/{num_epochs} " f"Loss: {average_loss:.4f} " f"LR: {current_lr:.4f}")

    torch.save(model.state_dict(),"./models/model.pth")


if __name__ == "__main__":
    main()
