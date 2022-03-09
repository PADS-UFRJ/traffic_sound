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

from classFeatureExtractor import FeatureExtractor 
from dataset import Traffic_Dataset
from utils import *


# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

  # Verificando os caminhos onde os dados serão salvos
  path = PATH_EXTRACTED_FEATURES+ EXTRACTION_MODEL+'/'

  if not os.path.exists(path):
    os.makedirs(path)


  # Abrindo arquivo com as principais informaçõs da extração 
  file_report = open(path+'Relatorio_geral.txt','w+')
  file_report.write('\n******* Extração de features *******\n\n')
  file_report.write('\nDispositivo usado pelo Pytorch: {}\n'.format(device))
  file_report.close()

  
  # Informações sobre a gpu usada
  if device.type == 'cuda':
    gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu)
    file_report = open(path+'Relatorio_geral.txt','a')
    file_report.write('\nGPU : {} ({})\n\n'.format(gpu,gpu_name))
    file_report.close()
  
  version_pytorch = torch.__version__

  file_report = open(path+'Relatorio_geral.txt','a')
  file_report.write('\nVersao biblioteca torch={}\n'.format(version_pytorch))
  file_report.write('\nRede convolucional usada:{}\n'.format(EXTRACTION_MODEL))
  file_report.write('\nInício da execução do codigo: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
  file_report.close()
  

  # Inicializando o modelo da vgg-16
  if (EXTRACTION_MODEL == 'vgg16'):
    model = models.vgg16(pretrained=True)


  # Inicializando o novo modelo apenas com as camadas convolucionais
  new_model = FeatureExtractor(model)
  
  
  if torch.cuda.device_count() > 1:
    new_model = nn.DataParallel(new_model,device_ids = [1, 0, 2, 3])


  # Enviando o modelo para a gpu 
  new_model.to(f'cuda:{new_model.device_ids[0]}')
  

  # Loop de extracao para cada video
  for video in range(VIDEOS_NUMBER):

    # Carregando o numero de frames que cada video possui 
    frames_tensor = np.load('dataset/' + videos_list[video] + '/imagedata.npy')
    number_frames = frames_tensor.shape[0]
    

    # Instanciando o objeto dataset através da classe Traffic_Dataset
    dataset = Traffic_Dataset(videos_list[video]) 


    # Inicializando arrays numpy e tensores
    frames_array = np.array([])
    frames_array_v = np.array([])
    features_array = torch.tensor([])
    pressure = np.array([])


    # Empilhando os frames e pressões sonoras
    for index in range(number_frames):
      frame_sample,pressure_sample = dataset[index]  
      frames = frame_sample.detach().numpy()
      frames_array = np.append(frames_array,frames)
      pressure = np.append(pressure_sample,pressure)
    

    # Remodelando o array numpy de frames em um tensor torch do tipo float no formato : (number_frames,3,224,224)
    frames_array = np.reshape(frames_array,(number_frames,3,224,224)) 
    frames_array = torch.from_numpy(frames_array).float() 
    

    # Instanciando o objeto DataLoader
    dataloader = DataLoader(frames_array,batch_size = BATCH_SIZE,shuffle=False)
    

    # Iterador usado para iterar entre os lotes de frames  
    dataloader_iter = iter(dataloader)


    # Inicializando o indice relacionado à quantidade de frames de um video
    frames_index = 0 


    with open(path+'Relatorio_geral.txt',"a") as file_report:
      file_report.write('\n-----> Video {}:{}\n'.format(video,videos_list[video]))
      file_report.write('\nNumero de frames: {}\n'.format(number_frames))
      file_report.write('\nBatch size:{}\n'.format(BATCH_SIZE))
      file_report.write('\nInicio da extracao : {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      
    
    # Loop para realizar a extração em batches
    for batch_index in range(number_frames//BATCH_SIZE):
      
      # next() retorna o próximo lote de frames 
      frames = next(dataloader_iter) 
      frames = frames.to(device)


      # Atualiando o indice relacionado à quantidade de frames de um video
      frames_index = frames_index + len(frames)


      # Extraindo as features e enviando para a cpu
      features = new_model(frames)
      features = features.to('cpu')
      features = features.detach().numpy()
      features_array = np.append(features_array,features)
      

     # Verificação caso a divisão dos dados em batches não seja inteira.
    if (frames_index < number_frames):

      verification_index = frames_index + 1
      while(verification_index <= number_frames):
        
        frame_sample_v,pressure_sample_v = dataset[verification_index]  
        frames_v = frame_sample_v.detach().numpy()
        frames_array_v = np.append(frames_array_v,frames_v)

        verification_index+=1

      frames_array_v = np.reshape(frames_array_v,((number_frames-frames_index),3,224,224)) 
      frames_array_v = torch.from_numpy(frames_array_v).float()

      features = new_model(frames_array_v)
      features = features.to('cpu')
      features = features.detach().numpy()
      features_array = np.append(features_array,features)
      features_array = np.reshape(features_array,(number_frames,512))


    # Salvando a matriz de features
    np.save(path +'Features_'+ videos_list[video],features_array)


    # Salvando o vetor com as pressoes sonoras
    np.save(path +'Sound-Pressure_'+ videos_list[video],features_array)
    
  
    with open(path+'Relatorio_geral.txt',"a") as file_report:
      file_report.write('\nTermino da extracao: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      file_report.write('\nFormato matriz de features: {}\n'.format(features_array.shape))
      file_report.write('\nFormato do vetor de pressões sonoras: {}\n'.format(pressure.shape))
        
    
  with open(path+'Relatorio_geral.txt',"a") as file_report:
    file_report.write('\n\nTermino da execução do codigo: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
  
    
    