import numpy as np
import os
from shutil import copy
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
from models import ModelFromDict

from train import train
from test import test

import argparse
import json

if not torch.cuda.is_available():
    raise Exception('GPU not available')

# Parseando argumentos
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('model', help='The JSON file describing the model to be trained')
# parser.add_argument('extractor', help='The name of the feature extractor model')
# parser.add_argument('features_offline', help='Indicates wether the features were extracted offline previously')
parser.add_argument('--gpu', default='0', help='The index of the GPU on which to run the train routine')
parser.add_argument('params', help='The JSON file containing the train hyperparameters')
parser.add_argument('-d', '--date', default='today')

###############################################################################

def parse_model_config(model_json, hyperparams):

    with open(model_json) as json_file:
        model_config = json.load(json_file)

    layers_dict = model_config['layers']
    # [TODO] colocar opcao de outros extratores de feature
    layers_dict['feature_extractor'] = {
                                        'model': layers_dict['feature_extractor'],
                                        'offline': True if ('features_offline' not in hyperparams) \
                                                    else hyperparams['features_offline']
                                    }

    layers_dict['fc_network']['dropout_value'] = None if ('fc_dropout' not in hyperparams) \
                                                else hyperparams['fc_dropout']

    return model_config


# Função que retorna o otimizador 
def optimizer_config(opt, model_params, value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model_params, lr = value_lr)
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model_params, lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model_params, lr = value_lr)
    return optimizer


def parse_hyperparams(hyperparams_json):

    with open(hyperparams_json) as json_file:
        hyperparams = json.load(json_file)

    # Listas necessárias para a busca por parametros
    epochs     = hyperparams['epochs']     if isinstance(hyperparams['epochs']    , list) else [ hyperparams['epochs'] ]
    optimizer  = hyperparams['optimizer']  if isinstance(hyperparams['optimizer'] , list) else [ hyperparams['optimizer'] ]
    batch_size = hyperparams['batch_size'] if isinstance(hyperparams['batch_size'], list) else [ hyperparams['batch_size'] ]
    fc_dropout = hyperparams['fc_dropout'] if isinstance(hyperparams['fc_dropout'], list) else [ hyperparams['fc_dropout'] ] # dropout 0 não dá certo !
    lr         = hyperparams['lr']         if isinstance(hyperparams['lr']        , list) else [ hyperparams['lr'] ]

    permutation = list(itertools.product(epochs, optimizer, batch_size, fc_dropout, lr))
    permutation = pd.DataFrame(permutation, columns=['epochs', 'optimizer', 'batch_size', 'fc_dropout', 'lr'])

    return permutation

if __name__ == '__main__':

    args = parser.parse_args()
    gpu = args.gpu
    today = args.date if (args.date != 'today') else datetime.now().strftime("%y%m%d")

    device = torch.device('cuda:' + gpu)

    permutation = parse_hyperparams(args.params)

    for i in range(len(permutation)):
        # df = pd.DataFrame(list(zip(permutation[i])), columns = ['n'])
        hyperparams = permutation.iloc[i]

        # Setando a seed do pytorch e do numpy !
        # [TODO] passar semente como um hiperparametro
        torch.manual_seed(22)
        np.random.seed(22)

        # Retornando o modelo
        model_config = parse_model_config(model_json=args.model, hyperparams=hyperparams)

        model = ModelFromDict(model_config)
        if len(permutation) > 1:
            model.name = model.name + f'_{i}'
        print(model)
        model = model.to(device)

        # Função de perda
        loss_function = nn.MSELoss()


        # Otimizador 
        optimizer = optimizer_config(hyperparams['optimizer'], model.parameters(), hyperparams['lr'])

        results_df = pd.DataFrame([ i for i in range(hyperparams['epochs']) ], columns=['epoch'])
        results_df.set_index('epoch')


        RESULTS_DIR = os.path.join(const.RESULTS_DIR, today, model.name)

        if not os.path.isdir( RESULTS_DIR ):
            os.makedirs( RESULTS_DIR )


        # salvando o modelo e os hiperparametros de treino em um JSON para facilitar o acesso posteriormente
        json_path = os.path.join(RESULTS_DIR, 'hyperparams.json')
        # with open(json_path, 'w') as f:
        #     json.dump(hyperparams, f, indent=4)
        hyperparams.to_json(json_path, indent=4)

        json_path = os.path.join(RESULTS_DIR, 'model.json')
        with open(json_path, 'w') as f:
            copy(args.model, json_path)

        # Loop de treino para os 10 folds
        for fold in folds_list:
            print(f'----> Fold {fold["index"]}')

            model.reset_parameters()
            # for layers in model.children():
            #     for layer in layers:
            #         if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            #             layer.reset_parameters()
            #         else:
            #             print(f"Nao possui reset_parameters: {layer}")

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

                    feature_extractor = model_config['layers']['feature_extractor']['model']
                    features_files[mode].append( os.path.join(const.FEATURES_DIR, feature_extractor, video + '_features.npy') )
                    targets_files[mode].append( os.path.join(const.TARGETS_DIR, video + '_targets.npy') )

                print(f'Inicializando datasets ({mode})')
                print('Carregando arquivos de features...')
                features_datasets[mode] = FeaturesDataset(features_files[mode]) # O dataset que retorna as features de acordo com a lista de arquivos
                print('Carregando arquivos de targets...')
                targets_datasets[mode] = AudioTargetDataset(targets_files[mode]) # O dataset que retorna os targets de acordo com a lista de arquivos

                # O dataset que une features e targets para serem retornados em uma tupla
                full_datasets[mode] = FeaturesAndTargetsUnionDataset(features_datasets[mode], targets_datasets[mode])

                # [NOTE] por algum motivo, o batch_size era passado como numpy.int64 e o DataLoader nao aceitava
                hyperparams['batch_size'] = int(hyperparams['batch_size'])
                dataloaders[mode] = DataLoader(full_datasets[mode], batch_size=hyperparams['batch_size'], shuffle=True )

            
            train_loss_list = []
            test_loss_list = []
            time_list = []
            min_test_loss = np.inf

            print('Iniciando treino')

            for epochs_index in range(hyperparams['epochs']):

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
                # checkpoint_save_path = os.path.join(RESULTS_DIR, fold['name'] + '_checkpoint.pth')
                checkpoint_save_path = os.path.join(RESULTS_DIR, 'checkpoints', fold['name'] + '_checkpoint.pth')
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    torch.save(model.state_dict(), checkpoint_save_path)

                print(f'Epoch: {epochs_index+1}\t Train Loss: {round(train_loss,4)} \t Test Loss: {round(test_loss,4)} \t (Best: {round(min_test_loss,4)}) \t [{round(time_list[-1], 1)}s]')


            # model_save_path = os.path.join(RESULTS_DIR, fold['name'] + '_model.pth')
            model_save_path = os.path.join(RESULTS_DIR, 'model_params', fold['name'] + '_model.pth')
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
            plt.title(model.name+'/'+fold['name'])
            plt.legend()
            # plt.savefig( os.path.join(RESULTS_DIR, fold["name"] + '_plot.png') )
            plt.savefig( os.path.join(RESULTS_DIR, 'plots', fold["name"] + '_plot.png') )

            results_df[ fold['name'] + '_trn'] = results_df[ fold['name'] + '_trn'].round(4)
            results_df[ fold['name'] + '_val'] = results_df[ fold['name'] + '_val'].round(4)
            results_df[ fold['name'] + '_time'] = results_df[ fold['name'] + '_time'].round(3)

            # Salvando os resultados de todos os folds
            # (o arquivo eh sobrescrito em cada fold. Eh redundante, porem nao precisamos esperar o treino de todos os folds terminar
            # para poder conferir o resultado dos que ja tiverem encerrado)
            results_csv_path = os.path.join( RESULTS_DIR, 'history.csv' )
            results_df.to_csv( results_csv_path, index=False )