# Importando as bibliotecas 
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import time
from datetime import datetime
from PIL import Image

import torch
from torch import optim, nn
from torchvision import models, transforms

from classVggPytorch import FeatureExtractor 
from datasetPytorch import Traffic_Dataset
from utils import *

# Definindo a GPU que será usada para a execução do código
os.environ["CUDA_VISIBLE_DEVICES"] = FEBE_GPU_NUMBER

# Função que retorna a tupla de dados no formato de lista para o dataloader
def collate_fn(dataset):
    x = dataset
    zipped = zip(x)
    return list(zipped)

# Função principal
if __name__ == '__main__':

  # Caminhos onde os dados serão salvos
  path = PATH_EXTRACTED_FEATURES+ EXTRACTION_MODEL+'/'

  if not os.path.exists(path):
    os.makedirs(path)

  # Abrindo arquivo com as principais informaçõs da extração 
  file = open(path+'Relatorio_geral.txt','w+')
  file.write('\n******* Extração de features *******\n')
  file.write('Versao bibliotecas :torch={} e python={}.\n'.format(TORCH_VERSION,PYTHON_VERSION))
  file.write('Rede convolucional usada:{}\n'.format(EXTRACTION_MODEL))
  file.close()

  # Inicializando o modelo da vgg-16
  if (EXTRACTION_MODEL == 'vgg16'):
    model = models.vgg16(pretrained=True)
  
  # Inicializando o novo modelo apenas com as camadas convolucionais
  new_model = FeatureExtractor(model)

  print("Inicio ao loop de extracao para cada video\n")

  # Loop de extracao para cada video
  for video in range(VIDEOS_NUMBER):
    print("VIDEO:{}".format(video))
    
    print("Carregando o numero de frames.\n")
    # Carregando o numero de frames que cada video possui 
    frames_tensor = np.load('dataset/' + videos_list[video] + '/imagedata.npy')
    number_frames = frames_tensor.shape[0]
    


    # Instanciando o objeto dataset através da classe Traffic_Dataset
    dataset = Traffic_Dataset(videos_list[video])
    print("Inicio ao retorno dos frames e pressoes de cada video\n")
    # Frames e presões sonoras de cada video
    frames,pressure = dataset[number_frames]
   
    print("Criando o dataloader\n")
    # Instanciando o DataLoader
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=False, collate_fn=collate_fn)
    
    # Pegando o tamanho do dataset 
    dataset_size = len(dataset)
    
    # Inicializando o vetor que salvará as features
    features_array = np.array([])

    dataloader_iter = iter(dataloader)

    # Inicializando o indice relacionado à quantidade de frames de um video
    frames_index = 0 


    with open(path+'Relatorio_geral.txt',"a") as file:
      file.write('\n-----> Video {}:{}\n'.format(video,videos_list[video]))
      file.write('Numero de frames: {}\n'.format(number_frames))
      file.write('Batch size:{}\n'.format(BATCH_SIZE))
      file.write('Inicio da extracao : {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    with open(path+'Relatorio_' + videos_list[video] +'.txt',"w+") as file:
      file.write('\n-----> Video:{}\n'.format(videos_list[video]))
      file.write('Batch size:{}\n'.format(BATCH_SIZE))
      file.write('Numero de iteracoes: {}\n\n'.format(dataset_size//BATCH_SIZE)) 

    print("Inicio ao loop para realizar a extracao em batches\n")

    # Loop para realizar a extração em batches
    for batch_index in range(dataset_size//BATCH_SIZE):
      try:
        with open(path+'Relatorio_' + videos_list[video] +'.txt',"a") as file:
          file.write('\tBatch:{}\n'.format(batch_index))
        
        # Frames é uma lista de tensores contendo os frames e as pressões sonoras.
        # Caso meu batch seja de tamanho 7 , teremos uma lista de 7 tuplas no formato (frame,pressão)
        frames = next(dataloader_iter) 
        
        # Loops necessáros para iterar pelas imagens em cada batch
        for count,tensor in enumerate(frames):
          #print("tensor:{}".format(type(tensor)))
          #print("count: {}".format(count))
          for frames_list_index,data_tuple in enumerate(frames[count]):
            frames_inside_batch,pressure_inside_batch = data_tuple
          
          # Indice de verificação 
          verification_index = frames_index
          
          # Extração das features
          if (frames_index == (dataset_size-1) and count == (len(frames))-1):
            features = new_model(frames_inside_batch)
            features = features.detach().numpy()
            features_array = np.append(features_array,features)
            frames_index = -1
          frames_index+=1  
        
      except StopIteration:
        dataloader_iter = iter(dataloader)
    
    # Verificação caso a divisão dos dados em batches não seja inteira.
    while(verification_index < dataset_size and frames_index):
      if verification_index == (dataset_size-1):
        frames_inside_verification ,pressure_inside_verification = dataset[verification_index]
        features_inside_verification = new_model(frames_inside_verification )
        features_inside_verification = features_inside_verification.detach().numpy()
        features_array = np.append(features_array,features_inside_verification)
      verification_index+=1

    features_array = np.reshape(features_array,((dataset_size-1),512))

    print("Salvando a matriz de features.\n")

    # Salvando a matriz de features
    np.save(path +'Features_'+ videos_list[video],features_array)

    with open(path+'Relatorio_geral.txt',"a") as file:
      file.write('Termino da extracao: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      file.write('\nFormato matriz de features: {}\n'.format(features_array.shape))

    
    
    