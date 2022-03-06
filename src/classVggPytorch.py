# Importando as bibliotecas 
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

import torch
from torch import optim, nn
from torchvision import models, transforms

from utils import *

# Definindo a GPU que será usada para a execução do código
os.environ["CUDA_VISIBLE_DEVICES"] = FEBE_GPU_NUMBER

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

