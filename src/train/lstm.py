from distutils.log import error
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
#from functions import *

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

        self.dense = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_value)
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first = True,
                            dropout=self.dropout_value_lstm,
                            bidirectional=self.bidirectional)

    def forward(self, x):

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
       
        x = self.dropout(hn)
        x = self.dense(x)
        #print(x)
        return x     

# Dataset Lstm
class LSTM_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self,folds,fold_number,mode,overlap,causal,n_steps):
        '''Define os valores iniciais.'''
        self.m = mode
        self.fold_number = fold_number
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []

        
        for video_index in folds[self.fold_number][self.m]:
            print(video_index)
            # Meus arquivos de features     
            #training_frames = np.load(os.path.join(PATH_FEATURES_CAROL,'Features_'+video_index+'.npy'))
            #training_targets = np.load(os.path.join(PATH_FEATURES_CAROL,'Sound-Pressures_'+ video_index +'.npy'))
            
            # Arquivos do felipe 
            #training_frames = np.load(os.path.join(PATH_FEATURES_FELIPE + video_index +'_features.npy'))
            #training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
            #training_targets = np.mean(training_targets, axis=1)

            # Arquivo para o uso dos targets do matheus
            training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
            training_targets = np.mean(training_targets, axis=1)

            #self.frames_list.append(training_frames)
            self.pressures_list.append(training_targets)
        """
        for i in self.frames_list:
            for j in range(i.shape[0]):
                self.list_of_all_frames.append(i[j])
        """
        for i in self.pressures_list:
            for j in range(i.shape[0]):
                self.list_of_all_pressures.append(i[j])
        print(f'len pressure list {len(self.list_of_all_pressures)}') 
        #self.frames_array = np.array(self.list_of_all_frames)
        self.pressures_array = np.array(self.list_of_all_pressures)
        
        # features-train do matheus
        path_matheus = '/home/mathlima/dataset/folds/vgg16/'
        if self.m == 'train':
            training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_train_input_data_gap.npy')
        else:
            training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_test_input_data_gap.npy')
    
        print(f'shape trainining {training_frames.shape}')    

        self.frames_array = torch.from_numpy(training_frames).float()
        self.pressures_array = torch.from_numpy(self.pressures_array).float()
        print(f'shape trainining again {self.frames_array.shape}')    

        #self.data_len = len(self.list_of_all_frames)
        self.data_len = len(self.list_of_all_pressures) 
        print(self.data_len)
        

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
            print('aq')
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
        #print(X)
        
        self.frames_array = X
        self.pressures_array = y
        print(f' pres  {len(self.pressures_array)}')
        self.len = len(self.frames_array) 
        print(self.len)
         

    def __getitem__(self, index): # indice do janela escolhida e do target correspondente a janela
        '''Retorna o item de número determinado pelo indice'''

        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''
        
        # Retorno o numero de janelas
        return self.len

# Função de treino 
def train(model,train_dataset,loss_function,optimizer):
    model.train()
    train_loss = 0.0

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=OPTION_SHUFFLE,num_workers=OPTION_NUM_WORKERS)
   
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
        """
        # Pegando as predições e as pressões na ultima época 
        if last_epoch == True:
            pred = pred.cpu()
            pred_numpy = pred.detach().numpy()
            list_of_predictions.append(pred_numpy)
            pressure = pressure.cpu()
            pressures_numpy = pressure.detach().numpy()
            list_of_pressures.append(pressures_numpy)
            last_epoch_model_path = os.path.join(path,'last_epoch_model/')

            if not os.path.exists(last_epoch_model_path):
                os.makedirs(last_epoch_model_path)

            # Salvando o modelo 
            torch.save(model.state_dict(),last_epoch_model_path+'model_last_epoch_fold_'+str(fold_index)+'.pth')
        """
    return (train_loss/len(train_dataset))
    
# Função de validação 
def validation(model,val_dataset,loss_function):
    model.eval()
    val_loss = 0.0
    min_val_loss = np.inf

    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=OPTION_SHUFFLE,num_workers=OPTION_NUM_WORKERS)

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
            """
            if min_val_loss > val_loss:
                #print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                min_val_loss = val_loss
                best_model_saved_path = os.path.join(path,'best_model/')

                if not os.path.exists(best_model_saved_path):
                    os.makedirs(best_model_saved_path)

                # Salvando o modelo 
                torch.save(model.state_dict(),best_model_saved_path+'best_model_fold_'+str(fold_index)+'.pth')
            """
        return (val_loss/len(val_dataset))

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


# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


training_results_path = os.path.join("results/",datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"/")
    
if not os.path.exists(training_results_path):
    os.makedirs(training_results_path)

file_name = 'Training_report-'+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') +'.txt'

history_file = open(training_results_path+file_name,'w+')
history_file.write('\nDispositivo usado pelo Pytorch: {}\n'.format(device))
history_file.close()

# Setando a seed do pytorch e do numpy !
torch.manual_seed(22)
np.random.seed(22)

num_epochs = 100
dropout = 0.5
lr = 1e-4
dropout_value_lstm = 0.2

history_file = open(training_results_path+file_name,'a')
history_file.write('\n OPTION_SHUFFLE={}\n\n'.format(OPTION_SHUFFLE))
history_file.write('\n num_epochs={} , dropout={} ,lr={}\n\n'.format(num_epochs,dropout,lr))
history_file.write('\n LSTM ={}\n\n'.format(LSTM))
history_file.write('\n option_overlap={} , option_causal={} , dropout_value_lstm={}\n\n'.format(option_overlap,option_causal,dropout_value_lstm))
history_file.close()


# Loop de treino e validação para os 10 folds
for fold_index in range(folds_number):

    print(f'----> Fold {fold_index}')
    
    history_file = open(training_results_path+file_name,'a')
    history_file.write('\n\n--> Fold: {}\n'.format(fold_index))
    history_file.close()

    train_dataset = LSTM_Dataset(folds,         # passamos o dicionário de folds
                                            fold_index,     # o indice do fold
                                            'train',        # o modo que queremos : 'train'
                                            option_overlap, # se tem sobreposição de janelas ou não
                                            option_causal,  # se é causal ou não
                                            size_windows)   # o tamanho da janela                                                                                                                                                                                                                                                    
   
    
    val_dataset = LSTM_Dataset( folds,         # passamos o dicionário de folds
                                            fold_index,     # o indice do fold
                                            'val',          # o modo que queremos : 'val'
                                            option_overlap, # se tem sobreposição de janelas ou não
                                            option_causal,  # se é causal ou não
                                            size_windows)   # o tamanho da janela                                                                                                                                                                                                                                                    
    
    
    model = LSTM_Network(INPUT_SIZE_FEATURES,OUTPUT_SIZE_FEATURES,HIDDEN_SIZE,dropout,num_layers,dropout_value_lstm,bidirectional) 

    print(model)
    model = model.to(device)

    # Função de perda
    loss_function = nn.MSELoss()

    # Otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
    
    list_loss_train = []
    list_loss_val = []
    min_val_loss = np.inf

    begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    history_file = open(training_results_path+file_name,'a')
    history_file.write('\nBegin of training: {}'.format(begin))
    history_file.write('\n\nEpoch,train_loss,val_loss,lr')
    history_file.close()


    for epochs_index in range(num_epochs):

        train_loss = train(model,train_dataset,loss_function,optimizer)
    
        val_loss = validation(model,val_dataset,loss_function)
        
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            epoch_of_min_val_loss = epochs_index+1

        history_file = open(training_results_path+file_name,'a')    
        history_file.write('\n{},{},{},{}'.format(epochs_index+1,train_loss,val_loss,lr))
        history_file.close()

        list_loss_train.append(train_loss)
        list_loss_val.append(val_loss)

        print(f'Epoch: {epochs_index+1}\t Training Loss: {train_loss} \t Val Loss: {val_loss}')

    end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Curva de treino 
    df_train = pd.DataFrame(list_loss_train, columns = ['train_loss'])
    loss_min = df_train['train_loss'].values.min()

    df_val = pd.DataFrame(list_loss_val, columns = ['val_loss'])
    val_loss_min = df_val['val_loss'].values.min()

    training_files_path = os.path.join(training_results_path,'files_training/')
        
    if not os.path.exists(training_files_path):
        os.makedirs(training_files_path)

    df_train.to_csv(training_files_path+"train_loss_fold_"+str(fold_index)+".csv", columns = ['train_loss'])
    df_val.to_csv(training_files_path+"val_loss_fold_"+str(fold_index)+".csv", columns = ['val_loss'])

    graphic_of_training(df_train,df_val,fold_index,training_files_path,num_epochs)

    history_file = open(training_results_path+file_name,'a')
    history_file.write('\nEnd of training: {}\n'.format(end))
    history_file.write('\nLoss min training: {}\n'.format(loss_min))
    history_file.write('\nLoss min val: {}\n'.format(val_loss_min))
    history_file.write('\nBest epoch: {}\n'.format(epoch_of_min_val_loss))
    history_file.close()  
    



