# Importando as bibliotecas 
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import torch
from torch import optim, nn
from torchvision import models, transforms

from utils import *

# Definindo a GPU que será usada para a execução do código
os.environ["CUDA_VISIBLE_DEVICES"] = FEBE_GPU_NUMBER


class Traffic_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, videos_list):
        '''Define os valores iniciais.'''
        

    def __getitem__(self, index):
        '''Retorna o item de número determinado pelo indice''' 
        self.size_data = index
        
        frames_index = 0
        video_index = 0
        auxiliary = 0
        
        frames_array = np.array([])
        sound_pressure = np.array([])

        preprocess = transforms.Compose([  
                    transforms.Resize(size=224),      
                    transforms.ToTensor(),       
                    transforms.Normalize(         
                    mean=[0.485, 0.456, 0.406],   
                    std=[0.229, 0.224, 0.225])])

        while(frames_index != index):
            
            frames_tensor = np.load('dataset/' + videos_list[video_index] + '/imagedata.npy')
            
            auxiliary = auxiliary + frames_tensor.shape[0]

            audio_array =  np.load('dataset/' + videos_list[video_index] + '/audioData.npy')
            audio_array = np.mean(audio_array, axis=1) # Media da posicao 1 
            
            while (frames_index != auxiliary) and (frames_index != index):

                image = Image.open('dataset/' + videos_list[video_index] + '/'+str(frames_index) +'.png')    
                image_preprocess = preprocess(image)
                frames_array = np.append(frames_array,image_preprocess)

                sound_pressure = np.append(sound_pressure,audio_array[frames_index])
                       
                frames_index+=1            
            video_index+=1
        
        frames_array = torch.from_numpy(frames_array).float()
        frames_array = np.reshape(frames_array,(index,3,224,224))

        return frames_array,sound_pressure

    def __len__(self):
        '''Número total de amostras'''
        return self.size_data 