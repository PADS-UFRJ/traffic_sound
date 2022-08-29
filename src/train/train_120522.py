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

from utils import *
#from Myfolds import *
from Myfolds_120522 import *

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')


# Arquitetura da rede FC
class Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers_size_list,dropout_value):
        super(Network, self).__init__()

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

# Dataset  que retorna a tupla (frames,pressoes) de 1 fold
class Folds_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,fold_number,mode):
        '''Define os valores iniciais.'''
        self.m = mode
        self.fold_number = fold_number
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []
        
        for i in folds[self.fold_number][self.m]:
            
            # Meus arquivos de features     
            #training_frames = np.load(path +'Features_'+ i +'.npy')
            #training_targets = np.load(path +'Sound-Pressures_'+ i +'.npy')
            
            # Arquivos do felipe 
            training_frames = np.load(PATH_FEATURES_FELIPE + i +'_features.npy')
            training_targets = np.load(PATH_TARGETS_FELIPE + i +'_targets.npy')
            training_targets = np.mean(training_targets, axis=1)

            # Arquivo para o uso dos targets do matheus
            #training_targets = np.load('/home/mathlima/dataset/' + i +'/output_targets.npy')
            #training_targets = np.mean(training_targets, axis=1)

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

        self.frames_array = torch.from_numpy(self.frames_array).float()
        self.pressures_array = torch.from_numpy(self.pressures_array).float()

        #print(f'shape frames in dataset {frames_array.shape}')
        #print(f'shape pressure in dataset {pressures_array.shape}')

        self.data_len = len(self.list_of_all_frames)  

    def __getitem__(self, index): # indice do fold escolhido e modo de treino ou teste
        '''Retorna o item de número determinado pelo indice'''

        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''

        return self.data_len

# Função de treino 
def train(model,dataloader,loss_function,optimizer,last_epoch):
    model.train()
    train_loss = 0.0
    list_of_predictions = []
    list_of_pressures = []
    for frames,pressure in dataloader:        
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
        train_loss += loss.item()

        # Pegando as predições e as pressões na ultima época 
        if last_epoch == True:
            pred = pred.cpu()
            pred_numpy = pred.detach().numpy()
            list_of_predictions.append(pred_numpy)
            pressure = pressure.cpu()
            pressures_numpy = pressure.detach().numpy()
            list_of_pressures.append(pressures_numpy)
        
    return ((train_loss/len(dataloader)),list_of_predictions,list_of_pressures)
    
# Função de teste 
def test(model,dataloader,loss_function,path,fold_index):
    model.eval()
    test_loss = 0.0
    min_test_loss = np.inf
    with torch.no_grad():
        for frames,pressure in dataloader:
            
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
                best_model_saved_path = path + '/' + 'best_model' + '/' 

                if not os.path.exists(best_model_saved_path):
                    os.makedirs(best_model_saved_path)

                # Salvando o modelo 
                torch.save(model.state_dict(),best_model_saved_path+'best_model_fold_'+str(fold_index)+'.pth')
        return test_loss/len(dataloader)

# Função que retorna o otimizador 
def optimizer_config(opt,value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr = value_lr)
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr = value_lr)
    return optimizer

# Função que plota gráfico da curva de treino 
def graphic_of_training(df_train,df_test,fold_index,path,epochs_number,train_loss_max):
    df_train.plot(ax=plt.gca())
    df_test.plot(ax=plt.gca())
    limiar = 4 #train_loss_max*0.25
    plt.axis([0,(epochs_number+1),0,limiar])
    plt.title('Loss_plot-Fold_'+str(fold_index))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+'Loss_plot-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que plota gráfico com a curva de treino de todos os folds 
def graphic_of_training_all_folds(df_train_all_folds,df_val_all_folds,path):
    print('aq2')
    mean_val = pd.DataFrame(columns=['val_loss_mean'])
    mean_train = pd.DataFrame(columns=['train_loss_mean'])

    mean_val['val_loss_mean'] = df_val_all_folds.mean(axis=1)
    mean_train['train_loss_mean'] = df_train_all_folds.mean(axis=1)

    mean_val.plot(color='violet',alpha=1.0,ax=plt.gca())
    mean_train.plot(color='pink',alpha=1.0,ax=plt.gca())
    
    # iterando entre as colunas do data frame de validação
    for column in df_val_all_folds:
        df_column = pd.DataFrame(val_list[column].values)
        df_column.plot(color='violet',alpha=0.4,ax=plt.gca())

    plt.title('Loss_plot-All_Folds')
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
    predict_path = path+'predictions_of_each_video_in_fold_'+str(fold_index)+'/'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    plt.title('Predicition-Fold_'+str(fold_index))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Predicition-Fold_'+str(fold_index)+'-'+video_index+'.png')
    plt.clf()

# Listas necessárias para a busca por parametros
epochs = [100]
opt = ['adam']
batch = [32]
dropout = [0.5] # dropout 0 não dá certo !
lr = [1e-4]


if __name__ == '__main__':

    
    path = PATH_EXTRACTED_FEATURES+ EXTRACTION_MODEL+'/'

    
    permutation = list(itertools.product(epochs,opt,batch,dropout,lr))

    training_results_path = 'results/'+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') +'/'
    
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)
    
    file_name = 'Training_report-'+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') +'.txt'

    history_file = open(training_results_path+file_name,'w+')
    history_file.write('\nDispositivo usado pelo Pytorch: {}\n'.format(device))
    history_file.close()

    train_list = pd.DataFrame()
    val_list = pd.DataFrame()

    for permutation_index in range(len(permutation)):
        df = pd.DataFrame(list(zip(permutation[permutation_index])), columns = ['n'])
    
         
        history_file = open(training_results_path+file_name,'a')
        history_file.write('\n*****************************************\n')
        history_file.write('\n-----> GRID: {}\n'.format(df))
        history_file.close()
            

        # Setando a seed do pytorch e do numpy !
        torch.manual_seed(22)
        np.random.seed(22)

    
        # Loop de treino e validação para os 10 folds
        for fold_index in range(folds_number):
        #for fold_index in range(2):
            print(f'----> Fold {fold_index}')
            
            history_file = open(training_results_path+file_name,'a')
            history_file.write('\n\n--> Fold: {}\n'.format(fold_index))
            history_file.close()

            # Dados de treino 
            
            train_dataset = Folds_Dataset(folds,fold_index,mode='train') # passamos o dicionário de folds 
                                                                         # e o indice do fold 
                                                                         # e o modo que queremos : 'train'

            train_frames_array = []
            train_pressures_array = []
            
            len_train = len(train_dataset)

            for index in range(len_train):
                train_frame,train_pressure = train_dataset[index] 
                train_frames_array.append(train_frame)
                train_pressures_array.append(train_pressure)
            
            # features-train do matheus
            #path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            #train_frames_array = np.load(path_matheus+'fold_'+ str(fold_index)+'_train_input_data_gap.npy')
            
            data_train = list(zip(train_frames_array,train_pressures_array))
            
            train_loader = DataLoader(train_dataset,batch_size = df['n'][2],shuffle=True,num_workers=3)#,collate_fn=collate_fn)
            
            
            # Dados de teste(validação!)
            test_dataset = Folds_Dataset(folds,fold_index,mode='test') # passamos o dicionário de folds 
                                                       # e o modo que queremos : 'test'
            
            len_test = len(test_dataset)

            test_frames_array = []
            test_pressures_array = []
            
            for index in range(len_test):
                test_frame,test_pressure = test_dataset[index] 
                test_frames_array.append(test_frame)
                test_pressures_array.append(test_pressure)
            
            # features-test do matheus 
            #path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            #test_frames_array = np.load(path_matheus+'fold_'+ str(fold_index)+'_test_input_data_gap.npy')
            
            data_test = list(zip(test_frames_array,test_pressures_array))
            
            test_loader = DataLoader(test_dataset,batch_size = df['n'][2],shuffle=True,num_workers=3)
            
            
            history_file = open(training_results_path+file_name,'a')
            history_file.write('\nFormat of train data: frames: {} , pressures: {} \n'.format(len(data_train),len(data_train)))
            history_file.write('\nFormat of test data: frames: {} , pressures: {} \n'.format(len(data_test),len(data_test)))
            history_file.close()

            # Retornando o modelo 
            model = Network(512,1,[128],df['n'][3])
            print(model)
            model = model.to(device)
            #summary(model, (len_train,512))

            # Função de perda
            loss_function = nn.MSELoss()


            # Otimizador 
            optimizer = optimizer_config(df['n'][1],df['n'][4])
            #scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

            loss_min_train = []
            loss_min_test = []
            
            begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            history_file = open(training_results_path+file_name,'a')
            history_file.write('\nBegin of training: {}'.format(begin))
            history_file.write('\n\nEpoch,train_loss,test_loss,lr')
            history_file.close()
        
            last_epoch = False
            predictions = []
            pressures = []
            min_test_loss = np.inf

            for epochs_index in range(df['n'][0]):

                train_loss,null_list,null_list = train(model,train_loader,loss_function,optimizer,last_epoch)
                if epochs_index == (int(df['n'][0])-1):
                    last_epoch = True
                    train_loss,predictions,pressures = train(model,train_loader,loss_function,optimizer,last_epoch)

                test_loss = test(model,test_loader,loss_function,training_results_path,fold_index)
                #scheduler.step()
                
                if min_test_loss > test_loss:
                    min_test_loss = test_loss
                    epoch_of_min_test_loss = epochs_index+1

                history_file = open(training_results_path+file_name,'a')    
                history_file.write('\n{},{},{},{}'.format(epochs_index+1,train_loss,test_loss,df['n'][4]))
                history_file.close()

                loss_min_train.append(train_loss)
                loss_min_test.append(test_loss)

                print(f'Epoch: {epochs_index+1}\t Training Loss: {train_loss} \t Test Loss: {test_loss}')

            
            end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Curva de treino 
            df1 = pd.DataFrame(loss_min_train, columns = ['train_loss'])
            print(f'df1 {df1}')
            loss_min = df1['train_loss'].values.min()

            df2 = pd.DataFrame(loss_min_test, columns = ['test_loss'])
            print(f'df2 {df2}')
            test_loss_min = df2['test_loss'].values.min()

            graphic_of_training(df1,df2,fold_index,training_results_path,df['n'][0],df1['train_loss'].values.max())
        
            history_file = open(training_results_path+file_name,'a')
            history_file.write('\nEnd of training: {}\n'.format(end))
            history_file.write('\nLoss min training: {}\n'.format(loss_min))
            history_file.write('\nLoss min test: {}\n'.format(test_loss_min))
            history_file.write('\nBest epoch: {}\n'.format(epoch_of_min_test_loss))
            history_file.close()  
        
            # Gráfico de predição
            list_of_all_pressures = []
            list_of_all_predictions = []

            print(f'len predictions {len(predictions)}')
            for i in predictions:
                for j in range(i.shape[0]):
                    list_of_all_predictions.append(i[j])
            print(f'len list all predictions {len(list_of_all_predictions)}')
            
            list_of_all_predictions = np.array(list_of_all_predictions)
            list_of_all_predictions = np.array(list_of_all_predictions,dtype= 'float64')
            list_of_all_predictions = np.squeeze(list_of_all_predictions)

            for i in pressures:
                for j in range(i.shape[0]):
                    list_of_all_pressures.append(i[j])
            
            list_of_all_pressures = np.array(list_of_all_pressures)
            pressure_npy = np.array(list_of_all_pressures,dtype= 'float64')
            pressure_npy = np.squeeze(pressure_npy)

            # Plotando o gráfico de predição para cada video dentro de 1 fold
            begin_index = 0
            auxiliary = 0
            for i in folds[fold_index]['train']:
                
                training_targets = np.load(PATH_TARGETS_FELIPE + i +'_targets.npy')
                number_frames =  training_targets.shape[0] 
                auxiliary = auxiliary + number_frames
                
                df_pressure = pd.DataFrame(pressure_npy[begin_index:auxiliary],columns = ['Real_samples'])
                df_prediction = pd.DataFrame(list_of_all_predictions[begin_index:auxiliary],columns = ['Predicted_samples'])
            
                graphic_of_video_predictions(df_pressure,df_prediction,fold_index,i,training_results_path)
                
                predict_files_path = training_results_path+'files_predictions'+'/'+'fold_'+str(fold_index)+'/'
                
                if not os.path.exists(predict_files_path):
                    os.makedirs(predict_files_path)

                np.save(predict_files_path + 'predictions_'+ i,pressure_npy[begin_index:auxiliary])

                begin_index = auxiliary


            # Calculando a correlação

            correlation = pearsonr(pressure_npy,list_of_all_predictions)
            print(f'type a {type(correlation)} {correlation}')
            
            history_file = open(training_results_path+file_name,'a')
            history_file.write('\nCorrelação: {}\n\n\n'.format(correlation))
            history_file.close()
            
            # Curva de treino de todos os folds

            # Adicionando os dados de treino e validação nos data frames 
            val_list.loc[:,'test_loss_'+str(fold_index)]=df2['test_loss']
            train_list.loc[:,'train_loss_'+str(fold_index)]=df1['train_loss']
        
        # Plotando a curva de treino de todos os folds
        print('aq1')
        graphic_of_training_all_folds(train_list,val_list,training_results_path) 
        
        

        
        
       
    
        

         
            
            
        
