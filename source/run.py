import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import const
from folds import folds as folds_list
from dataset_classes import FeaturesDataset, AudioTargetDataset, FeaturesAndTargetsUnionDataset
from models import FCNetwork

from train import train
from test import test


if not torch.cuda.is_available():
    raise Exception('GPU not available')

gpu = '1'
device = torch.device('cuda:' + gpu)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# Função que retorna o otimizador 
def optimizer_config(opt, value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr = value_lr)
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr = value_lr)
    return optimizer


# Listas necessárias para a busca por parametros
epochs = [100]
opt = ['adam']
batch = [32]
dropout = [0.5] # dropout 0 não dá certo !
lr = [1e-4]

feature_extractor = 'vgg16'

if __name__ == '__main__':

    permutation = list(itertools.product(epochs, opt, batch, dropout, lr))

    for i in range(len(permutation)):
        df = pd.DataFrame(list(zip(permutation[i])), columns = ['n'])

        # Setando a seed do pytorch e do numpy !
        torch.manual_seed(22)
        np.random.seed(22)

        # Retornando o modelo 
        model = FCNetwork(512,1,[128],df['n'][3])
        print(model)
        model = model.to(device)

        
        # Função de perda
        loss_function = nn.MSELoss()


        # Otimizador 
        optimizer = optimizer_config(df['n'][1],df['n'][4])

        results_df = pd.DataFrame([ i for i in range(df['n'][0]) ], columns=['epoch'])
        results_df.set_index('epoch')

        # today = '220425'
        today = datetime.now().strftime("%y%m%d")
        if not os.path.isdir( os.path.join(const.RESULTS_DIR, today, feature_extractor) ):
            os.makedirs( os.path.join(const.RESULTS_DIR, today, feature_extractor) )

        const.RESULTS_DIR = os.path.join(const.RESULTS_DIR, today, feature_extractor)

        # Loop de treino para os 10 folds
        for fold in folds_list:
            print(f'----> Fold {fold["index"]}')

            model.apply(reset_weights)

            # Inicializando dicionarios vazios. A chave indica se o elemento retornado eh usado no treino ou no teste
            features_files = {'train': None, 'test': None}
            targets_files = {'train': None, 'test': None}

            features_datasets = {'train': None, 'test': None}
            targets_datasets = {'train': None, 'test': None}

            full_datasets = {'train': None, 'test': None}

            dataloaders = {'train': None, 'test': None}

            for mode in ['train', 'test']:

                features_files[mode] = [] # Lista de todos os arquivos de features correspondentes ao fold
                targets_files[mode] = [] # Lista de todos os arquivos de targets correspondentes ao fold

                for video in fold[mode]:

                    # !>[TODO] Apagar data
                    features_files[mode].append( os.path.join(const.FEATURES_DIR, feature_extractor, video + '_features.npy') )
                    targets_files[mode].append( os.path.join(const.TARGETS_DIR, video + '_targets.npy') )

                print(f'Inicializando datasets ({mode})')
                print('Carregando arquivos de features...')
                features_datasets[mode] = FeaturesDataset(features_files[mode]) # O dataset que retorna as features de acordo com a lista de arquivos
                print('Carregando arquivos de targets...')
                targets_datasets[mode] = AudioTargetDataset(targets_files[mode]) # O dataset que retorna os targets de acordo com a lista de arquivos

                # O dataset que une features e targets para serem retornados em uma tupla
                full_datasets[mode] = FeaturesAndTargetsUnionDataset(features_datasets[mode], targets_datasets[mode])

                dataloaders[mode] = DataLoader(full_datasets[mode], batch_size=df['n'][2], shuffle=True )

            
            train_loss_list = []
            test_loss_list = []
            time_list = []
            min_test_loss = np.inf

            print('Iniciando treino')

            for epochs_index in range(df['n'][0]):

                # begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                begin = time.time()
                
                train_loss = train(model, dataloaders['train'], loss_function, optimizer, device)
                
                # end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                train_loss_list.append(train_loss)

                test_loss = test(model, dataloaders['test'], loss_function, device)

                test_loss_list.append(test_loss)

                end = time.time()

                time_list.append( end - begin )

                # Salvando o modelo de melhor loss
                checkpoint_save_path = os.path.join(const.RESULTS_DIR, fold['name'] + '_checkpoint.pth')
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    torch.save(model.state_dict(), checkpoint_save_path)

                print(f'Epoch: {epochs_index+1}\t Train Loss: {round(train_loss,4)} \t Test Loss: {round(test_loss,4)} \t (Best: {round(min_test_loss,4)})')


            model_save_path = os.path.join(const.RESULTS_DIR, fold['name'] + '_model.pth')
            torch.save(model.state_dict(), model_save_path)

            results_df[ fold['name'] + '_trn'] = train_loss_list
            results_df[ fold['name'] + '_val'] = test_loss_list
            results_df[ fold['name'] + '_time'] = time_list
            val_min = results_df[ fold['name'] + '_val'].min()


            # df1 = pd.DataFrame(train_loss_list, columns = ['train_loss'])
            # loss_min = df1['train_loss'].values.min()

            plt.clf()
            plt.plot(train_loss_list, label='trn', color=const.colors[0])
            plt.plot(test_loss_list, label='val', color=const.colors[1])
            plt.axhline(y=val_min, color='r', linestyle='--')
            plt.ylim(bottom=0, top=3)
            plt.title(feature_extractor+'/'+fold['name'])
            plt.legend()
            plt.savefig( os.path.join(const.RESULTS_DIR, fold["name"] + '_plot.png') )

            results_df[ fold['name'] + '_trn'] = results_df[ fold['name'] + '_trn'].round(4)
            results_df[ fold['name'] + '_val'] = results_df[ fold['name'] + '_val'].round(4)
            results_df[ fold['name'] + '_time'] = results_df[ fold['name'] + '_time'].round(3)

            # Salvando os resultados de todos os folds
            # (o arquivo eh sobrescrito em cada fold. Eh redundante, porem nao precisamos esperar o treino de todos os folds terminar
            # para poder conferir o resultado dos que ja tiverem encerrado)
            results_csv_path = os.path.join( const.RESULTS_DIR, 'history.csv' )
            results_df.to_csv( results_csv_path, index=False )