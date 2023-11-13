import argparse
from pathlib import Path
import json
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import const
from models import ModelFromDict
from dataset_classes import FeaturesDataset, AudioTargetDataset, FeaturesAndTargetsUnionDataset, LSTM_Dataset
from folds import test_videos
from run import parse_model_config, parse_hyperparams
from validation import validate

parser = argparse.ArgumentParser()
parser.add_argument('model_results_dir', help='The directory containing results from the specific model train run.')
parser.add_argument('--gpu', default='0', help='The index of the GPU on which to run the train routine')


if __name__ == '__main__':

    args = parser.parse_args()
    model_dir = Path(args.model_results_dir).resolve()
    device = torch.device('cuda:' + args.gpu)

    permutation, _ = parse_hyperparams(model_dir/'hyperparams.json')
    hyperparams = permutation.iloc[0] # nao deve haver mais de uma combinacao de hiperparametros
    model_config = parse_model_config(model_dir/'model.json', hyperparams=hyperparams, device=device)

    model = ModelFromDict(model_config)
    # [?] model = model.to(device)

    extractor = model_config['layers']['feature_extractor']['model']

    features = [ Path(const.FEATURES_DIR)/extractor/f'{video}_features.npy' for video in test_videos ]
    targets = [ Path(const.TARGETS_DIR)/f'{video}_targets.npy' for video in test_videos ]

    features = FeaturesDataset([ str(path) for path in features ])
    targets = AudioTargetDataset([ str(path) for path in targets ])

    if 'model_61' in model.name:
        dataset = LSTM_Dataset(features, targets, overlap=True, causal=False, n_steps=32)
    else:
        dataset = FeaturesAndTargetsUnionDataset(features, targets)    

    # [?] precisamos de shuffle = True?
    dataloader = DataLoader(dataset, batch_size=int(hyperparams['batch_size']), shuffle=False)
    loss_function = nn.MSELoss()

    results = pd.DataFrame({'fold': [], 'test_loss': []})

    for checkpoint in (model_dir/'checkpoints').iterdir():

        fold = checkpoint.name.replace('_checkpoint.pth', '') # ex: fold_6_checkpoint.pth
        fold = {'name': fold, 'index': int(fold.split('_')[-1])}

        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        model.to(device)
        model.eval()

        loss_value = validate(model, dataloader, loss_function, device)

        results = results.append({'fold': fold['index'], 'test_loss': loss_value}, ignore_index=True)

    results['fold'] = results['fold'].astype(int)
    print(f'Salvando arquivo: {model_dir/"test_results.csv"}')
    results.sort_values(by='fold').to_csv(model_dir/'test_results.csv', index=False)
    print("Arquivo salvo.")

