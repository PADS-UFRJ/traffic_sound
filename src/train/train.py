from distutils.log import error
import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import random
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from utils import *
from Myfolds import *
from functions import *

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu
device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    if(LSTM == True):
        permutation = list(itertools.product(epochs,opt,batch,dropout,lr,dropout_lstm))
    else:
        permutation = list(itertools.product(epochs,opt,batch,dropout,lr))
    
    
    time_file = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    file_name = 'Training_report-'+ time_file     

    train_list = pd.DataFrame()
    val_list = pd.DataFrame()

    # Loop do Grid Search
    for permutation_index in range(len(permutation)):
        df = pd.DataFrame(list(zip(permutation[permutation_index])),columns = ['hyperparameters'])
        
        epochs_grid = df['hyperparameters'][0]
        optmizer_grid = df['hyperparameters'][1]
        batch_grid = df['hyperparameters'][2]
        dropout_grid = df['hyperparameters'][3]
        lr_grid = df['hyperparameters'][4]
        if(LSTM == True):
            dropout_lstm_grid = float(df['hyperparameters'][5])
            print(type(dropout_lstm_grid))
        
        training_results_path = os.path.join("results/",time_file+"/Grid_"+str(permutation_index)+"/")
        
        if not os.path.exists(training_results_path):
            os.makedirs(training_results_path)
        
        # Salvando a versão dos códigos de treino.
        if (permutation_index == 0):
            # O 'if' é necessário quando temos mais um ponto do grid,pois caso eu faça alguma alteração em um dos códigos ,
            # irei salvar apenas a primeira versão, que foi a versão usada para executar o grid.
            save_current_version_of_codes(time_file)
        
        file_path = training_results_path+file_name+"-Grid_"+str(permutation_index)+'.txt'

        history_file = open(file_path,'w+')
        history_file.write('\nDispositivo usado pelo Pytorch: {}\n'.format(device))
        history_file.write('\nFeatures usadas: {}\n'.format(FEATURES))
        history_file.write('\nSeed number {}\n'.format(SEED_NUMBER))
        history_file.close()

        history_file = open(file_path,'a')
        history_file.write('\n*****************************************\n')
        history_file.write('\n-----> GRID {}: {}\n'.format(permutation_index,df))
        history_file.close()
            
        
        if(LSTM == True):
            history_file = open(file_path,'a')
            history_file.write('\n LSTM ={}\n\n'.format(LSTM))
            history_file.write('\n option_overlap={} , option_causal={} ,option_dropout_lstm={}\n\n'.format(OPTION_OVERLAP,OPTION_CAUSAL,dropout_lstm_grid))
            history_file.close()

        list_val_loss_min = []

        # Loop de treino e validação para os 10 folds
        for fold_index in range(folds_number):
            print(f'----> Fold {fold_index}')
            
            history_file = open(file_path,'a')
            history_file.write('\n\n--> Fold: {}\n'.format(fold_index))
            history_file.close()

            # Setando a seed do pytorch,numpy e do python !
            torch.manual_seed(SEED_NUMBER)
            np.random.seed(SEED_NUMBER)
            random.seed(SEED_NUMBER)
            
            # Dados de treino 

            if(LSTM == True):
                print("train dataset lstm")
                train_dataset = LSTM_Dataset(folds,         # passamos o dicionário de folds
                                             fold_index,     # o indice do fold
                                             'train',        # o modo que queremos : 'train'
                                             OPTION_OVERLAP, # se tem sobreposição de janelas ou não
                                             OPTION_CAUSAL,  # se é causal ou não
                                             SIZE_WINDOWS)   # o tamanho da janela                                                                                                                                                                                                                                                    

            else:
                print("train dataset vgg")
                train_dataset = VGG_Dataset(folds,          # passamos o dicionário de folds
                                            fold_index,     # e o indice do fold 
                                            mode='train')   # e o modo que queremos : 'train'
                                                                            
                                                                            
            
            train_frames_array = [] # é uma lista com as features(vetor) 
            train_frames_tensor = [] # usado apanas na lstm ,é um lista com as janelas
            train_pressures_array = []
            val_frames_array = []
            val_pressures_array = []
             
            
            len_train = len(train_dataset)
            
            print(f' len train {len_train}')
            
            
            for index in range(len_train):
                train_frame,train_pressure = train_dataset[index]
                
                if(LSTM == True) and (OPTION_CAUSAL == False):
                    train_frames_array.append(train_frame[SIZE_WINDOWS//2 - 1])
                    train_frames_tensor.append(train_frame)

                elif(LSTM == True) and (OPTION_CAUSAL == True):
                    train_frames_array.append(train_frame[SIZE_WINDOWS - 1])
                    train_frames_tensor.append(train_frame)
                else: # vgg !
                    train_frames_array.append(train_frame)
                
                train_pressures_array.append(train_pressure)
                
            # Dados de validação
            if(LSTM == True):
                print("val dataset lstm")
                val_dataset = LSTM_Dataset( folds,         # passamos o dicionário de folds
                                            fold_index,     # o indice do fold
                                            'val',          # o modo que queremos : 'val'
                                            OPTION_OVERLAP, # se tem sobreposição de janelas ou não
                                            OPTION_CAUSAL,  # se é causal ou não
                                            SIZE_WINDOWS)   # o tamanho da janela                                                                                                                                                                                                                                                    
            else:
                print("val dataset vgg")
                val_dataset = VGG_Dataset(folds,        # passamos o dicionário de folds
                                          fold_index,   # e o indice do fold 
                                          mode='val')   # e o modo que queremos : 'val'  
                    

            len_val = len(val_dataset)
            
            for index in range(len_val):
                val_frame,val_pressure = val_dataset[index] 

                if(LSTM == True) and (OPTION_CAUSAL == False):
                    val_frames_array.append(val_frame[SIZE_WINDOWS//2 - 1])

                elif(LSTM == True) and (OPTION_CAUSAL == True):
                    val_frames_array.append(val_frame[SIZE_WINDOWS - 1])
                
                else:
                    val_frames_array.append(val_frame)
                
                val_pressures_array.append(val_pressure)
            
            
            history_file = open(file_path,'a')
            history_file.write('\nFormat of train data: frames: {} , pressures: {} \n'.format(len(train_frames_array),len(train_pressures_array)))
            history_file.write('\nFormat of validation data: frames: {} , pressures: {} \n'.format(len(val_frames_array),len(val_pressures_array)))
            history_file.write('\nOption shuffle: {}\n'.format(OPTION_SHUFFLE))
            history_file.close()

            # Retornando o modelo 
            if(LSTM == True):
                model = LSTM_Network(INPUT_SIZE_FEATURES,OUTPUT_SIZE_FEATURES,HIDDEN_SIZE,dropout_grid,NUM_LAYERS,dropout_lstm_grid,BIDIRECTIONAL) 
            else:
                model = VGG_Network(INPUT_SIZE_FEATURES,OUTPUT_SIZE_FEATURES,[128],dropout_grid)
            
            print(model)
            model = model.to(device)
            

            # Função de perda
            loss_function = nn.MSELoss()


            # Otimizador 
            optimizer = optimizer_config(optmizer_grid,model,lr_grid)

            
            list_loss_train = []
            list_loss_val = []
            list_predictions = []
            list_of_all_predictions = []
            min_val_loss = np.inf

            begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            history_file = open(file_path,'a')
            history_file.write('\nBegin of training: {}'.format(begin))
            history_file.write('\n\nEpoch,train_loss,val_loss,lr')
            history_file.close()


            for epochs_index in range(epochs_grid):
                """
                # Otimizador
                # Se SCHEDULER = True, temos a lr variando de acordocom as épocas. Não usei a função StepLR do pytorch, pois, neste caso, o gamma varia!
                if (SCHEDULER):
                    if (epochs_index < NUMBER_STEPS_EPOCHS):
                        optimizer = optimizer_config(optmizer_grid,model,lr_scheduler[0])
                        lr_grid = lr_scheduler[0]

                    elif (epochs_index >= NUMBER_STEPS_EPOCHS and epochs_index <= NUMBER_STEPS_EPOCHS*2):
                        optimizer = optimizer_config(optmizer_grid,model,lr_scheduler[1])
                        lr_grid = lr_scheduler[1]

                    elif (epochs_index > NUMBER_STEPS_EPOCHS*2 and epochs_index < epochs_grid ):
                        optimizer = optimizer_config(optmizer_grid,model,lr_scheduler[2])
                        lr_grid = lr_scheduler[2]

                else:
                    # Se SCHEDULER = False, temos um lr constante para todas as épocas.
                    optimizer = optimizer_config(optmizer_grid,model,lr_grid)     
                """

                train_loss = train(model,train_dataset,loss_function,optimizer,batch_grid)
                
                val_loss = validation(model,val_dataset,loss_function,training_results_path,fold_index,batch_grid)
                
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    epoch_of_min_val_loss = epochs_index+1

                history_file = open(file_path,'a')    
                history_file.write('\n{},{},{},{}'.format(epochs_index+1,train_loss,val_loss,lr_grid))
                history_file.close()

                list_loss_train.append(train_loss)
                list_loss_val.append(val_loss)

                print(f'Epoch: {epochs_index+1}\t Training Loss: {train_loss} \t Val Loss: {val_loss}')
    
            end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Curva de treino 
            df_train = pd.DataFrame(list_loss_train, columns = ['train_loss'])
            train_loss_min = df_train['train_loss'].values.min()

            df_val = pd.DataFrame(list_loss_val, columns = ['val_loss'])
            val_loss_min = df_val['val_loss'].values.min()
            list_val_loss_min.append(val_loss_min)

            training_files_path = os.path.join(training_results_path,'files_training/')
                
            if not os.path.exists(training_files_path):
                os.makedirs(training_files_path)

            df_train.to_csv(training_files_path+"train_loss_fold_"+str(fold_index)+".csv", columns = ['train_loss'])
            df_val.to_csv(training_files_path+"val_loss_fold_"+str(fold_index)+".csv", columns = ['val_loss'])

            graphic_of_training(df_train,df_val,fold_index,training_files_path,epochs_grid)
        
            history_file = open(file_path,'a')
            history_file.write('\nEnd of training: {}\n'.format(end))
            history_file.write('\nLoss min training: {}\n'.format(train_loss_min))
            history_file.write('\nLoss min val: {}\n'.format(val_loss_min))
            history_file.write('\nBest epoch: {}\n'.format(epoch_of_min_val_loss))
            history_file.close()  

        
            try: 
                
                #last_epoch_model_path = os.path.join(training_results_path,'last_epoch_model/'+'model_last_epoch_fold_'+str(fold_index)+'.pth')
                best_epoch_model_path = os.path.join(training_results_path,'best_model/'+'best_model_fold_'+str(fold_index)+'.pth')
                model.load_state_dict(torch.load(best_epoch_model_path))
                model.to(device)
                model.eval()
                    
            except: 
                print('Erro em abrir arquivo!')
            else:
                list_predictions = []
                pred_loader = DataLoader(val_dataset,batch_size=batch_grid,shuffle=False,num_workers=OPTION_NUM_WORKERS)
                with torch.no_grad():
                    for frames,pressure in pred_loader:        
                        frames= frames.to(device)
                        #print(frames.shape)
                        
                        # Passando os dados para o modelo para obter as predições
                        pred = model(frames)
                        list_predictions.append(pred)
                   
            for batch_index in list_predictions:
                for prediction_index in range(batch_index.shape[0]):
                    list_of_all_predictions.append(batch_index[prediction_index])
            print(f'len list all predictions {len(list_of_all_predictions)}')
            
            list_of_all_predictions = np.array(list_of_all_predictions,dtype= 'float64')
            list_of_all_predictions = np.squeeze(list_of_all_predictions)
            
            val_pressures_array = np.array(val_pressures_array,dtype= 'float64')
            val_pressures_array = np.squeeze(val_pressures_array)

            # Calculando a correlação
            correlation = pearsonr(val_pressures_array ,list_of_all_predictions)
            
            history_file = open(file_path,'a')
            history_file.write('\nCorrelation: {}\n\n\n'.format(correlation))
            history_file.close()

            # Plotando as predições de 1 fold inteiro

            predict_files_path = os.path.join(training_results_path,'files_predictions/')
                
            if not os.path.exists(predict_files_path):
                os.makedirs(predict_files_path)

            np.save(predict_files_path +'predictions_fold_'+str(fold_index),list_of_all_predictions)

            df_pressure = pd.DataFrame(val_pressures_array,columns=['Real_samples_best_model'])
            df_prediction = pd.DataFrame(list_of_all_predictions,columns=['Predicted_samples_best_model'])
                    
            graphic_of_fold_predictions(df_pressure,df_prediction,fold_index,predict_files_path)

            
            if (LSTM == False):
                # Plotando o gráfico de predição para cada video dentro de 1 fold da VGG
                begin_index = 0
                auxiliary = 0
                for video_index in folds[fold_index]['val']:
                    training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE,video_index+'_targets.npy'))
                    training_targets = np.mean(training_targets, axis=1)
                    
                    number_frames =  training_targets.shape[0] 
                    auxiliary = auxiliary + number_frames

                    df_prediction['Predicted_samples_best_model'][begin_index:auxiliary]

                    predict_files_path_vgg = os.path.join(predict_files_path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
                    
                    if not os.path.exists(predict_files_path_vgg):
                        os.makedirs(predict_files_path_vgg)
                
                    graphic_of_video_predictions(df_pressure['Real_samples_best_model'][begin_index:auxiliary],
                                                df_prediction['Predicted_samples_best_model'][begin_index:auxiliary],
                                                fold_index,
                                                video_index,
                                                predict_files_path_vgg)
                    begin_index = auxiliary


                    
            # Curva de treino de todos os folds

            # Adicionando os dados de treino e validação nos data frames 
            val_list.loc[:,'val_loss_'+str(fold_index)]=df_val['val_loss']
            train_list.loc[:,'train_loss_'+str(fold_index)]=df_train['train_loss']
            
        # Plotando a curva de treino de todos os folds
        graphic_of_training_all_folds(train_list,val_list,training_files_path,epochs_grid) 
            
        history_file = open(file_path,'a')
        history_file.write('\n\nMean validation loss: {}\n'.format(sum(list_val_loss_min)/len(list_val_loss_min)))
        history_file.close()      

            



            

        

        
        
       
    
        

         
            
            
        
