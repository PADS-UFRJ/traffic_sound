import torch
from torch import nn

class VggFeatureExtractor(nn.Module):

    def __init__(self, vgg):
        super().__init__() # executamos a inicializacao da classe superior
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
            
    
    def forward(self, x):
        x = self.features(x)
        # print('features:', x.shape)

        x = self.avgpool(x)
        # print('avgpool:', x.shape)

        # x = self.maxpool(x)
        # print('maxpool:', x.shape)

        x = self.flatten(x)
        # print('flatten:', x.shape)
    
        return x 

