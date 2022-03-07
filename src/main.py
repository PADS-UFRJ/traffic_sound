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

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

  # Caminhos onde os dados serão salvos
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

  #model.to(device)


  # Inicializando o novo modelo apenas com as camadas convolucionais
  new_model = FeatureExtractor(model)
  

  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    new_model = nn.DataParallel(new_model,device_ids = [1, 0, 2, 3])
    #print("new model {}".format(new_model.device))
  #new_model.to(device)

  new_model.to(f'cuda:{new_model.device_ids[0]}')
  
  print("Inicio ao loop de extracao para cada video\n")

  # Loop de extracao para cada video
  for video in range(VIDEOS_NUMBER):
    print("VIDEO:{}".format(video))
    
    print("Carregando o numero de frames.\n")
    
    # Carregando o numero de frames que cada video possui 
    frames_tensor = np.load('dataset/' + videos_list[video] + '/imagedata.npy')
    number_frames = frames_tensor.shape[0]
    

    print("Numero frames {}".format(number_frames))

    # Instanciando o objeto dataset através da classe Traffic_Dataset
    dataset = Traffic_Dataset(videos_list[video]) 

    print("Inicio ao retorno dos frames e pressoes de cada video\n")

    frames = torch.tensor([])
    features_array = torch.tensor([])
    features_inside_verification = torch.tensor([])
    pressure = np.array([])

    for index in range(number_frames):
      frame_sample,pressure_sample = dataset[index]  
      frames = torch.cat([frame_sample, frames], dim=0)
      pressure = np.append(pressure_sample,pressure)
    
    """
    pressure = torch.as_tensor(np.array(pressure).astype('float'))
    pressure.to(device)
    torch.save(pressure,path +'Labels_'+ videos_list[video]+'.pt')
    """
    print("frames.shape {}".format(frames.shape))  
    
    
    # Instanciando o DataLoader
    dataloader = DataLoader(frames,batch_size = BATCH_SIZE,shuffle=False)
    
    # Pegando o tamanho do dataset 
    dataset_size = len(dataset)

    # Criando o iterador 
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
        
      print("\n------------- Batch {} ----------- \n".format(batch_index))

      
      frames = next(dataloader_iter) 
      #frames = frames.to(device)
      #frames = frames.to(f'cuda:{new_model.device_ids[0]}')
      
      print("frames {}".format(frames.device))
      print("Len frames {}".format(len(frames)))
      print("frames shape {}".format(frames.shape))
      
      features = new_model(frames)
      features = features.to(f'cuda:{new_model.device_ids[0]}')
      print("Features shape: {}".format(features.shape))
      #features = features.to(device)
      print("features {}".format(features.device))
      #features_array = features_array.to(device)
      features_array = features_array.to(f'cuda:{new_model.device_ids[0]}')
      features_array = torch.cat([features, features_array], dim=0)
      
     
    print(features_array.shape)
    
    verification_index = features_array.shape[0]
    
    # Verificação caso a divisão dos dados em batches não seja inteira.
    while(verification_index < number_frames):
      
      frames_inside_verification,pressure_inside_verification = dataset[verification_index]
      #frames_inside_verification = frames_inside_verification.to(device)
      #frames_inside_verification = frames_inside_verification.to(f'cuda:{new_model.device_ids[0]}')

      #features_inside_verification = features_inside_verification.to(device)
      features_inside_verification = new_model(frames_inside_verification )
      features_inside_verification = features_inside_verification.to(f'cuda:{new_model.device_ids[0]}')
      features_array = torch.cat([features_array,features_inside_verification], dim=0)
      
      verification_index+=1


    print(features_array.shape)
    #features_array = features_array.to(f'cuda:{new_model.device_ids[0]}')
    # Salvando a matriz de features
    #features_array=features_array.to(device)
    
    torch.save(features_array,path +'Features_'+ videos_list[video]+'.pt')

    with open(path+'Relatorio_geral.txt',"a") as file_report:
      file_report.write('\nTermino da extracao: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      file_report.write('\nFormato matriz de features: {}\n'.format(features_array.shape))
    break    
  
  with open(path+'Relatorio_geral.txt',"a") as file_report:
    file_report.write('\n\nTermino da execução do codigo: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
  
    
    