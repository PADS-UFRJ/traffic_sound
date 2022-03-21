import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import sqlite3
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import datasets

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

#from utils import *
from Myfolds import *

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
#device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class Network(nn.Module):
    def __init__(self,dropout_value):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_value)
        

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Dataset  que retorna a tupla (frames,pressoes) de 1 fold
class Folds_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,mode):
        '''Define os valores iniciais.'''
        self.m = mode
        
    def __getitem__(self, index): # indice do fold escolhido e modo de treino ou teste
        '''Retorna o item de número determinado pelo indice'''

        frames_list = []
        pressures_list = []
        list_of_all_frames = []
        list_of_all_pressures = []

        for i in folds[index][self.m]:
    
            training_frames = np.load(path +'Features_'+ i +'.npy')
            
            training_targets = np.load(path +'Sound-Pressures_'+ i +'.npy')

            frames_list.append(training_frames)
            pressures_list.append(training_targets)

        for i in frames_list:
            for j in range(i.shape[0]):
                list_of_all_frames.append(i[j])
        
        for i in pressures_list:
            for j in range(i.shape[0]):
                list_of_all_pressures.append(i[j])
        
        frames_array = np.array(list_of_all_frames)
        pressures_array = np.array(list_of_all_pressures)

        frames_array = torch.from_numpy(frames_array).float()
        pressures_array = torch.from_numpy(pressures_array).float()

        print(f'shape frames in dataset {frames_array.shape}')
        print(f'shape pressure in dataset {pressures_array.shape}')

        self.data_len = len(list_of_all_frames)
        
        return frames_array,pressures_array

    def __len__(self):
        '''Número total de amostras'''
        return self.data_len

# Função de treino 
def train(model,frames_dataloader,pressure_dataloader,loss_function,optimizer):
    model.train()
    train_loss = 0.0
    for frames,pressure in zip(frames_dataloader,pressure_dataloader):
            
        frames, pressure = frames.to(device), pressure.to(device)
        
        pressure_aux = pressure 
        pressure = pressure_aux[:,None]
        
        # Passando os dados para o modelo para obter as predições
        pred = model(frames)

        # Calculando a perda através da função de perda
        loss = loss_function(pred,pressure)

        # Zerando os gradientes acumulados.O Pytorch vai acumulando os gradientes de acordo com as operações realizadas .  
        optimizer.zero_grad()

        # Calculando os gradientes
        loss.backward()

        # Tendo o gradiente já calculado , o step indica a direção que reduz o erro que vamos andar 
        optimizer.step()

        # Loss é um tensor!
        train_loss += loss.item()
    return train_loss/len(frames_dataloader)
    
# Função de teste 
def test(model,frames_dataloader,pressure_dataloader,loss_function):
    model.eval()
    test_loss = 0.0
    min_test_loss = np.inf
    with torch.no_grad():
        for frames,pressure in zip(frames_dataloader,pressure_dataloader):
            
            frames, pressure = frames.to(device), pressure.to(device)
            
            pressure_aux = pressure 
            pressure = pressure_aux[:,None]
            
            # Passando os dados para o modelo para obter as predições
            pred = model(frames)

            # Calculando a perda através da função de perda
            loss = loss_function(pred,pressure)

            test_loss += loss.item()
            if min_test_loss > test_loss:
                #print(f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss:.6f}) \t Saving The Model')
                min_test_loss = test_loss
        
                # Salvando o modelo 
                torch.save(model.state_dict(), 'saved_model.pth')
        return test_loss/len(frames_dataloader)

# Função que retorna o otimizador 
def optimizer_config(opt,value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr = value_lr)
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr = value_lr)
    return optimizer

# Listas necessárias para a busca por parametros
epochs = [5]
opt = ['sgd']
batch = [32]
dropout = [0.2] # dropout 0 não dá certo !
lr = [1e-5]


if __name__ == '__main__':

    path = PATH_EXTRACTED_FEATURES+ EXTRACTION_MODEL+'/'

    permutation = list(itertools.product(epochs,opt,batch,dropout,lr))

    for i in range(len(permutation)):
        df = pd.DataFrame(list(zip(permutation[i])), columns = ['n'])

        # Setando a seed do pytorch e do numpy !
        torch.manual_seed(22)
        np.random.seed(22)

    
        # Loop de treino para os 10 folds
        for fold_index in range(folds_number):
            print(f'----> Fold {fold_index}')
            

            # Dados de treino 
            dataset = Folds_Dataset(folds,mode='train') # passamos o dicionário de folds 
                                                        # e o modo que queremos : 'train'

            train_data = dataset[fold_index] # indice do fold escolhido
            
            print(f'shape of train frames: {train_data[0].shape}')
            print(f'shape of train pressures:{train_data[1].shape}')
        
            train_frames_loader = DataLoader(train_data[0],batch_size = df['n'][2],shuffle=True )
            train_pressure_loader = DataLoader(train_data[1],batch_size = df['n'][2],shuffle=True )


            # Dados de teste
            dataset = Folds_Dataset(folds,mode='test') # passamos o dicionário de folds 
                                                       # e o modo que queremos : 'test'
            test_data = dataset[fold_index]

            print(f'shape of test frames: {test_data[0].shape}')
            print(f'shape of test pressures:{test_data[1].shape}')

            test_frames_loader = DataLoader(test_data[0],batch_size = df['n'][2],shuffle=True )
            test_pressure_loader = DataLoader(test_data[1],batch_size = df['n'][2],shuffle=True )
            

            # Retornando o modelo 
            model = Network(df['n'][3])
            print(model)
            model = model.to(device)
            

            # Função de perda
            loss_function = nn.MSELoss()


            # Otimizador 
            optimizer = optimizer_config(df['n'][1],df['n'][4])

            
            loss_min_list = []
            loss_min_test = []
            
            
            for epochs_index in range(df['n'][0]):
                begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                train_loss = train(model,train_frames_loader,train_pressure_loader,loss_function,optimizer)
                
                end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                loss_min_list.append(train_loss)

                test_loss = test(model,test_frames_loader,test_pressure_loader,loss_function)
                
                loss_min_test.append(test_loss)

                print(f'Epoch: {epochs_index+1}\t Training Loss: {train_loss} \t Test Loss: {test_loss}')

              
            
            df1 = pd.DataFrame(loss_min_list, columns = ['train_loss'])
            loss_min = df1['train_loss'].values.min()

            plt.plot(loss_min_list)
            plt.plot(loss_min_test)
            plt.savefig('loss.png')
        
        
        