import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax 

class FeatureExtractor(nn.Module):
  def __init__(self):
    super(FeatureExtractor, self).__init__()
    vgg_model = vgg19(pretrained=True)
    self.vgg = nn.Sequential(*list(vgg_model.children())[:18])
  def forward(self, x):
    return self.vgg(x)

class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(in_features, 0.8),
        nn.PReLU(),
        nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(in_features, 0.8)
    )
  
  def forward(self, x):
    return self.conv_block(x)

class GeneratorResNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, residual_blocks=16):
    super(GeneratorResNet, self).__init__()
    self.first_block = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

    res_blocks = []
    for _ in range(residual_blocks):
      res_blocks.append(ResidualBlock(64))
    self.res_blocks = nn.Sequential(*res_blocks)

    self.second_block = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

    upsampling = []
    for _ in range(2):
      upsampling += [nn.Conv2d(64, 256, kernel_size=3, stride=1),
                     nn.BatchNorm2d(256),
                     nn.PixelShuffle(upscale_factor=2),
                     nn.PReLU()]
    self.upsampling = nn.Sequential(*upsampling)

    self.final_blocks = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1))

  
  def forward(self, img):
    out_1 = self.first_block(img)
    out = self.res_blocks(out_1)
    out_2 = self.second_block(out)
    out = torch.add(out_1, out_2)
    out = self.upsampling(out)
    out = self.final_blocks(out)
    return out

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_feature = in_channels
        for i, out_filter in enumerate([64, 128, 256, 512]):
            layers.extend(self.make_block(in_feature, out_filter, stride=(i%2==0)))
            in_feature = out_filter

        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


    def make_block(self, in_features, out_features, stride=True):
        layers = []
        if stride:
            layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_features))
        layers.append(nn.LeakyReLU())
        return layers



model = Discriminator()
print(model)