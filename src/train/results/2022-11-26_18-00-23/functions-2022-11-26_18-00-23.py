from distutils.log import error
from re import X
import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import sqlite3
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable 

from utils import *
from Myfolds import *

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# Arquitetura da rede FC
class LSTM_Network(nn.Module):

    def __init__(self,input_size,output_size,hidden_size,dropout_value,num_layers,dropout_value_lstm,option_bidirectional):
        super(LSTM_Network, self).__init__() 

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_value = dropout_value
        self.num_layers = num_layers
        self.dropout_value_lstm = dropout_value_lstm
        self.bidirectional = option_bidirectional
        
        if self.bidirectional == True:
           self.num_directions = 2
        else:
            self.num_directions = 1 

        self.dense_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_linear_layer = nn.Dropout(self.dropout_value)
        self.dropout_lstm_layer = nn.Dropout(self.dropout_value_lstm)
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first = True,
                            dropout=self.dropout_value_lstm,
                            bidirectional=self.bidirectional)

    def forward(self, x):
        x = self.dropout_lstm_layer(x)

        # Inicializando os estados ocultos com tensores preenchidos por 0  
        h_0 = Variable(torch.zeros(self.num_layers*self.num_directions, x.size(0), self.hidden_size)) # hidden state
        c_0 = Variable(torch.zeros(self.num_layers*self.num_directions, x.size(0), self.hidden_size)) # internal state
        h_0,c_0 = h_0.to(device),c_0.to(device)
        #print(h_0.shape)

        # Alimentando a camada lstm com os estados ocultos e com os dados.
        #  
        # output = contém todos os estados ocultos ao longo da sequência.
        # (hn, cn) = tensor com os últimos estados da camada lstm

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # shape de hn -> [1,32,128]
        #print(hn.shape) 

        # Remodelando os dados para pode usar na camada dense 
        hn = hn.view(-1, self.hidden_size) # shape de hn -> [32,128]
        #print(hn.shape)

        x = torch.tanh(self.dense_hidden(hn))
        x = self.dropout_linear_layer(x)
        x = self.dense(x)
        #print(x)
        return x     


# Dataset Lstm
class LSTM_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,fold_number,mode,overlap,causal,n_steps):
        '''Define os valores iniciais.'''
        self.m = mode
        self.fold_number = fold_number
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []

        
        for video_index in folds[self.fold_number][self.m]:
            
            # Meus arquivos de features     
            #training_frames = np.load(os.path.join(PATH_FEATURES_CAROL,'Features_'+video_index+'.npy'))
            #training_targets = np.load(os.path.join(PATH_FEATURES_CAROL,'Sound-Pressures_'+ video_index +'.npy'))
            
            # Arquivos do felipe 
            training_frames = np.load(os.path.join(PATH_FEATURES_FELIPE + video_index +'_features.npy'))
            training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
            training_targets = np.mean(training_targets, axis=1)

            # Arquivo para o uso dos targets do matheus
            if (FEATURES == 'Matheus'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)

            self.frames_list.append(training_frames)
            self.pressures_list.append(training_targets)
        
        for i in self.frames_list:
            for j in range(i.shape[0]):
                self.list_of_all_frames.append(i[j])
        
        for i in self.pressures_list:
            for j in range(i.shape[0]):
                self.list_of_all_pressures.append(i[j])
         
        self.frames_array = np.array(self.list_of_all_frames)
        self.pressures_array = np.array(self.list_of_all_pressures)

        
        if (FEATURES == 'Matheus'):
            # features-train do matheus
            path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            if self.m == 'train':
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_train_input_data_gap.npy')
            else:
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_test_input_data_gap.npy')
            
            self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
        else:
            self.frames_array = torch.from_numpy(self.frames_array).float()

        
        self.pressures_array = torch.from_numpy(self.pressures_array).float()

        self.data_len = len(self.list_of_all_pressures) 
        #self.data_len = len(self.list_of_all_frames) 

        if (overlap == True) and (causal == True): 
            """
            w_frames_file = open('windows_with_overlap_causal.txt','w+')
            w_frames_file.write('overlap={} e causal={}\n'.format(overlap,causal))
            w_frames_file.close()
            """

            X = []
            y = []

            # Com sobreposição de janelas e de forma causal
            for i in range(self.data_len):

                end_ix = i + n_steps
            
                if end_ix > len(self.pressures_array)-1:
                    break

                seq_x, seq_y = self.frames_array[i:end_ix], self.pressures_array[end_ix-1]

                """
                w_frames_file = open('windows_with_overlap_causal.txt','a')
                w_frames_file.write('\n i {} end_ix {} - len seq_frames {}\n'.format(i,end_ix,len(seq_x)))
                w_frames_file.write(' {}\n'.format(seq_x))
                w_frames_file.write('target= {}\n'.format(seq_y))
                w_frames_file.close()
                """
                
                X.append(seq_x)
                y.append(seq_y)

        if (overlap == True) and (causal == False):
            print('aq2')
            """
            w_frames_file = open('windows_with_overlap_without_causal.txt','w+')
            w_frames_file.write('overlap={} e causal={}\n'.format(overlap,causal))
            w_frames_file.close()
            """

            X = []
            y = []
        
            # Com sobreposição de janelas e não causal
            for i in range(self.data_len):
                
                end_ix = i + n_steps
            
                if end_ix > len(self.pressures_array)-1:
                    break

                seq_x = self.frames_array[i:end_ix]
                target_index = i + (n_steps//2 - 1)
                seq_y = self.pressures_array[target_index]
            
                """
                w_frames_file = open('windows_with_overlap_without_causal.txt','a')
                w_frames_file.write('\n i {} end_ix {} - len seq_frames {}\n'.format(i,end_ix,len(seq_x)))
                w_frames_file.write(' {}\n'.format(seq_x))
                w_frames_file.write('target= {}\n'.format(seq_y))
                w_frames_file.close()
                """
                
                X.append(seq_x)
                y.append(seq_y)
        
        self.frames_array = X
        self.pressures_array = y
        self.len = len(self.frames_array)
         

    def __getitem__(self, index): # indice do janela escolhida e do target correspondente a janela
        '''Retorna o item de número determinado pelo indice'''

        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''
        
        # Retorno o numero de janelas
        return self.len


# Arquitetura da rede FC para a VGG16
class VGG_Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers_size_list,dropout_value):
        super(VGG_Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_size_list = hidden_layers_size_list
        size_current = input_size
        self.layers = nn.ModuleList()
        for size_index in hidden_layers_size_list:
            self.layers.append(nn.Linear(size_current, size_index))
            size_current = size_index
        self.layers.append(nn.Linear(size_current, output_size))
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        for layer in self.layers[:-1]: # Estou pegando todas as camadas,exceto a última 
            x = torch.tanh(layer(x))
        x = self.dropout(x)
        x = self.layers[-1](x)
        return x   
      

def initialize_weights(model):
    for layer in model.layers:
        nn.init.normal_(layer.weight)
        if layer.bias is not None:
            nn.init.normal_(layer.bias)
           

# Dataset  que retorna a tupla (frames,pressoes) de 1 fold
class VGG_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,fold_number,mode):
        '''Define os valores iniciais.'''
        self.mode = mode
        self.fold_number = fold_number
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []
        
        for video_index in folds[self.fold_number][self.mode]:
            
            # Meus arquivos de features     
            #training_frames = np.load(os.path.join(PATH_FEATURES_CAROL,'Features_'+video_index+'.npy'))
            #training_targets = np.load(os.path.join(PATH_FEATURES_CAROL,'Sound-Pressures_'+ video_index +'.npy'))
            
            # Arquivos do felipe 
            if (FEATURES == 'Felipe'):
                training_frames = np.load(os.path.join(PATH_FEATURES_FELIPE + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)
        
            # Arquivo para o uso dos targets do matheus
            if (FEATURES == 'Matheus'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)

            # Arquivos para as features extraidas carregando o modelo em pytorch e usando d vgg do tf/keras
            if (FEATURES == 'torch_model_with_weights_of_tf/keras'):
                training_frames = np.load(os.path.join(PATH_FEATURES_TF_KERAS + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)

            self.frames_list.append(training_frames)
            self.pressures_list.append(training_targets)

        for video_index in self.frames_list:
            for frame_index in range(video_index.shape[0]):
                self.list_of_all_frames.append(video_index[frame_index])
        
        for video_index in self.pressures_list:
            for pressure_index in range(video_index.shape[0]):
                self.list_of_all_pressures.append(video_index[pressure_index])
         
        self.frames_array = np.array(self.list_of_all_frames)
        self.pressures_array = np.array(self.list_of_all_pressures)

        if (FEATURES == 'Matheus'):
            # features-train do matheus
            path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            if self.mode == 'train':
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_train_input_data_gap.npy')
            else:
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_test_input_data_gap.npy')
            
            self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
        else:
            self.frames_array = torch.from_numpy(self.frames_array).float()

        
        self.pressures_array = torch.from_numpy(self.pressures_array).float()


        self.data_len = len(self.list_of_all_frames)  

    def __getitem__(self, index): # indice do fold escolhido e modo de treino ou validação
        '''Retorna o item de número determinado pelo indice'''

        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''

        return self.data_len

# Função de treino 
def train(model,train_dataset,loss_function,optimizer,batch_grid):
    model.train()
    train_loss = 0.0


    train_loader = DataLoader(train_dataset,batch_size=batch_grid,shuffle=OPTION_SHUFFLE,num_workers=OPTION_NUM_WORKERS)
   
    for frames,pressure in train_loader:        
        frames, pressure = frames.to(device), pressure.to(device)

        pressure_aux = pressure 
        pressures = pressure_aux[:,None]
        
        # Passando os dados para o modelo para obter as predições
        pred = model(frames)

        # Calculando a perda através da função de perda
        loss = loss_function(pred,pressures)

        # Zerando os gradientes acumulados.O Pytorch vai acumulando os gradientes de acordo com as operações realizadas .  
        optimizer.zero_grad()

        # Calculando os gradientes
        loss.backward()

        # Tendo o gradiente já calculado , o step indica a direção que reduz o erro que vamos andar 
        optimizer.step()

        # Loss é um tensor!
        train_loss += frames.size(0) * loss.item()
        
    return (train_loss/len(train_dataset))
    
# Função de validação 
def validation(model,val_dataset,loss_function,path,fold_index,batch_grid):
    model.eval()
    val_loss = 0.0
    min_val_loss = np.inf

    val_loader = DataLoader(val_dataset,batch_size=batch_grid,shuffle=OPTION_SHUFFLE,num_workers=OPTION_NUM_WORKERS)

    with torch.no_grad():
        for frames,pressure in val_loader:
            
            frames, pressure = frames.to(device), pressure.to(device)
            
            pressure_aux = pressure
            pressure = pressure_aux[:,None]
            
            # Passando os dados para o modelo para obter as predições
            pred = model(frames)

            # Calculando a perda através da função de perda
            loss = loss_function(pred,pressure)

            val_loss += frames.size(0) * loss.item()

            if min_val_loss > val_loss:
                #print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                min_val_loss = val_loss
                best_model_saved_path = os.path.join(path,'best_model/')

                if not os.path.exists(best_model_saved_path):
                    os.makedirs(best_model_saved_path)

                # Salvando o modelo 
                torch.save(model.state_dict(),best_model_saved_path+'best_model_fold_'+str(fold_index)+'.pth')
        return (val_loss/len(val_dataset))

# Função que retorna o otimizador 
def optimizer_config(opt,model,value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr = value_lr)
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr = value_lr)
    return optimizer

# Função que plota gráfico da curva de treino 
def graphic_of_training(df_train,df_val,fold_index,path,epochs_number):
    df_train.plot(ax=plt.gca())
    df_val.plot(ax=plt.gca())
    limiar = 4 #train_loss_max*0.25
    plt.axis([0,(epochs_number+1),0,limiar])
    plt.title('Loss_plot-Fold_'+str(fold_index))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+'Loss_plot-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que plota gráfico com a curva de treino de todos os folds 
def graphic_of_training_all_folds(df_train_all_folds,df_val_all_folds,path,epochs_grid):
    mean_val = pd.DataFrame(columns=['val_loss_mean'])
    mean_train = pd.DataFrame(columns=['train_loss_mean'])

    mean_val['val_loss_mean'] = df_val_all_folds.mean(axis=1)
    mean_train['train_loss_mean'] = df_train_all_folds.mean(axis=1)

    mean_val.plot(color='blue',alpha=1.0,ax=plt.gca(),legend=False)
    mean_train.plot(color='pink',alpha=1.0,ax=plt.gca(),legend=False)
    
    # iterando entre as colunas do data frame de validação
    for column in df_val_all_folds:
        df_column = pd.DataFrame(df_val_all_folds[column].values)
        df_column.plot(color='blue',alpha=0.4,ax=plt.gca(),legend=False)

    plt.title('Loss_plot-All_Folds')
    limiar = 5
    plt.axis([0,epochs_grid,0,limiar])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+'Loss_plot-All_Folds.png')
    plt.clf()

# Função que plota gráfico de predição de um fold  
def graphic_of_fold_predictions(df_pressures,df_prediction,fold_index,path):
    df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    plt.title('Predicition-Fold_'+str(fold_index))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(path+'Predicition-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que plota gráfico de predição de cada video  
def graphic_of_video_predictions(df_pressures,df_prediction,fold_index,video_index,path):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    plt.title('Predicition-Fold_'+str(fold_index))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Predicition-Fold_'+str(fold_index)+'-'+video_index+'.png')
    plt.clf()

# Função que plota gráfico de predição de cada fold  
def graphic_of_fold_predictions(df_pressures,df_prediction,fold_index,path):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    plt.title('Predicition-Fold_'+str(fold_index))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Predicition-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que salva o estado dos códigos no início do treino
def save_current_version_os_codes(time_file):
  
    source_train = "/home/caroline/traffic_sound/src/train/train.py"
    destination_train = "/home/caroline/traffic_sound/src/train/results/"+time_file+"/train.py"
    
    source_utils = "/home/caroline/traffic_sound/src/train/utils.py"
    destination_utils = "/home/caroline/traffic_sound/src/train/results/"+time_file+"/utils.py"

    source_functions = "/home/caroline/traffic_sound/src/train/functions.py"
    destination_functions = "/home/caroline/traffic_sound/src/train/results/"+time_file+"/functions.py"

    os.system('cp '+source_train+' '+destination_train)
    os.system('cp '+source_utils+' '+destination_utils)
    os.system('cp '+source_functions+' '+destination_functions)

    os.system('mv '+destination_train+' '+"/home/caroline/traffic_sound/src/train/results/"+time_file+"/train-"+time_file+".py")
    os.system('mv '+destination_utils+' '+"/home/caroline/traffic_sound/src/train/results/"+time_file+"/utils-"+time_file+".py")
    os.system('mv '+destination_functions+' '+"/home/caroline/traffic_sound/src/train/results/"+time_file+"/functions-"+time_file+".py")

