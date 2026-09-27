import torch
from torch import nn

class FastSRNet(nn.Module):
    def __init__(self,scale=2)->None:
        super().__init__()
        if scale !=2:
            raise ValueError("FastSRNet currently supports only scale=2")
        self.scale=scale
        self.model=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,  padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8,3*scale*scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)