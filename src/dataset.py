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



class Traffic_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, videos_list):
        '''Define os valores iniciais.'''
        
    def __len__(self):
        '''Número total de amostras'''
        return 1

    def __getitem__(self, index):
        '''Retorna o item de número determinado pelo indice''' 
        
        frames_index = 0
        video_index = 0
        auxiliary = 0
        
        find_index = False

        frames_array = np.array([])
        sound_pressure = np.array([])

        preprocess = transforms.Compose([  
                    transforms.Resize(size=224),      
                    transforms.ToTensor(),       
                    transforms.Normalize(         
                    mean=[0.485, 0.456, 0.406],   
                    std=[0.229, 0.224, 0.225])])

        while(find_index == False):

            audio_array =  np.load('dataset/' + videos_list[video_index] + '/audioData.npy')
            audio_array = np.mean(audio_array, axis=1) # Media da posicao 1 
            
            try:
                if video_index > 0:
                    new_index = index - auxiliary
                    image = Image.open('dataset/' + videos_list[video_index] + '/'+str(new_index) +'.png')    
                    pressure = audio_array[new_index]
                else:
                    image = Image.open('dataset/' + videos_list[video_index] + '/'+str(index) +'.png')    
                    pressure = audio_array[index]

            except (FileNotFoundError or IndexError) as error:
                
                if index > auxiliary:
                    frames_tensor = np.load('dataset/' + videos_list[video_index] + '/imagedata.npy')
                    auxiliary = auxiliary + frames_tensor.shape[0]
                    video_index+=1
                find_index = False
            
            else:
                find_index = True    
        
        frames_array = preprocess(image)
        frames_array = np.reshape(frames_array,(1,3,224,224))
        

        return frames_array,pressure

