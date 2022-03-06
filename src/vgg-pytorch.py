# Importando as bibliotecas 
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from PIL import Image

import torch
from torch import optim, nn
from torchvision import models, transforms


# Definindo a GPU que será usada para a execução do código
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# Carregando o tensor com as informações de cada frame (numero_frames, 224*224*3)
frames_tensor = np.load('dataset/M2U00001MPG/imagedata.npy')
print(frames_tensor.shape)
print(frames_tensor.dtype)


# Criando as transformaçoes que serão feitas em cada frame
preprocess =  transforms.Compose([          # Compose = Agrupa varias transformações
              transforms.ToTensor(),        # ToTensor = Converte um array numpy em tensor
              transforms.Normalize(         # Normalize = Normaliza o tensor com a média e o desvio padrão
              mean=[0.485, 0.456, 0.406],   # usados na ImageNet
              std=[0.229, 0.224, 0.225])])


# Inicializando um array numpy
frames_array = np.array([])


# Atribuindo alguns valores às variáveis 
#number_frames = frames_tensor.shape[0]
number_frames = 1
batch_size = 16
channels = 3


# Abrindo, pre-processando e empilhando cada frame de um vídeo
for index in range(number_frames):
  image = Image.open("dataset/M2U00001MPG/"+str(index)+".png")
  image_preprocess = preprocess(image)
  frames_array = np.append(frames_array,image_preprocess)
print(frames_array.shape)


# Transformando o array numpy em um tensor do tipo float
frames_array = torch.from_numpy(frames_array).float()


# Remodelando o formato do tensor para (numero_de_frames, canal , altura, largura)
frames_array = np.reshape(frames_array,(number_frames,channels,224,224)) 
print(frames_array.shape)
print(type(frames_array))


# Criando uma dimensão para o bach size 
frames_array = frames_array[:,None,:,:]
print(frames_array.shape)


# Classe que retorna o modelo para a extração de features 
class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    self.pooling = model.avgpool 
    self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
    self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
		
  
  def forward(self, x):
    out = self.features(x)
    print(out.shape)
    #out = self.pooling(out)
    print(out.shape)
    out = self.maxpool(out)
    print(out.shape)
    out = self.flatten(out)
    print(out.shape)
    
 
    return out 


# Inicializando o modelo da vgg-16
model = models.vgg16(pretrained=True)


# Inicializando o novo modelo apenas com as camadas para a extração
new_model = FeatureExtractor(model)
print(new_model)


# Inicializando um array numpy
features_array = np.array([])


# Extraindo as features imagem por imagem e empilhando no array numpy
for index in range(number_frames): 
  
  features = new_model(frames_array[index,:,:,:,:])
  #print(features.shape)

  features = features.detach().numpy()
  features_array = np.append(features_array,features)
  

print(features_array.shape)


# Remodelando o formato dos dados para o formato (numero_de_frames,features)
features_array = np.reshape(features_array,(number_frames,512)) 
print(features_array.shape)


"""
# Salvando as features
path = '/home/caroline/traffic-analysis/Teste-features-vgg16/'
name_file = "extracted_features_vgg16"

if not os.path.exists(path):
    os.makedirs(path)

np.save(path + name_file,features_array)
"""