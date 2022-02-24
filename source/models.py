import torch
from torch import nn

class VggFeatureExtractor(nn.Module):

    def __init__(self, vgg):
        super().__init__() # executamos a inicializacao da classe superior
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        # self.avgpool = vgg.avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
            
    
    def forward(self, x):
        out = self.features(x)
        # print('features:',out.shape)
        out = self.avgpool(out)
        # print('avgpool:',out.shape)
        out = self.maxpool(out)
        # print('maxpool:',out.shape)
        out = self.flatten(out)
        # print('flatten:',out.shape)
    
        return out 

