# Importando as bibliotecas 

import torch
from torch import optim, nn
from torchvision import models, transforms

from utils import *


# Classe que retorna o modelo para a extração de features 
class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features) 
    self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
    self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
		
  
  def forward(self, x):
    out = self.features(x)
    out = self.maxpool(out)
    out = self.flatten(out)
    
    return out 

