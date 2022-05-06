import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision as vision

import matplotlib.pyplot as plt

import const
from models import FCNetwork
from dataset_classes import FeaturesDataset

import numpy as np
import os
from os import path as pth

## -----------------------------------------------------

gpu = '1'
feature_extractor = 'vgg16'
date = '220505'
batch_size = 32

## -----------------------------------------------------

const.RESULTS_DIR = pth.join(const.RESULTS_DIR, date, feature_extractor)
const.FEATURES_DIR = pth.join(const.FEATURES_DIR, feature_extractor)

## -----------------------------------------------------

if not torch.cuda.is_available():
    raise Exception('GPU not available.')

device = torch.device('cuda:' + gpu)

## -----------------------------------------------------

model = FCNetwork(512, 1, [128], 0.2)

state_dict_filepath = pth.join(const.RESULTS_DIR, 'fold_0_model.pth')


state_dict = torch.load(state_dict_filepath, map_location='cpu')
model.load_state_dict(state_dict)

model.to(device)
model.eval()

## -----------------------------------------------------

features_files = os.listdir(const.FEATURES_DIR)
features_files.sort()
features_files = [ pth.join(const.FEATURES_DIR, f) for f in features_files ]

targets_files = os.listdir(const.TARGETS_DIR)
targets_files.sort()
targets_files = [ pth.join(const.TARGETS_DIR, f) for f in targets_files ]

## -----------------------------------------------------

dataset = FeaturesDataset( features_files[:1] )
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

predictions = []

with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)

        pred = model(batch)

        for p in pred:
            predictions.append( p.cpu().numpy()[0] )

predictions = np.array(predictions)
print(f'predictions.shape: {predictions.shape}')

## -----------------------------------------------------

targets = np.load(targets_files[0]).mean(axis=1)
print(f'targets.shape: {targets.shape}')

## -----------------------------------------------------

# m = predictions.mean()
# s = np.std(predictions)
# predictions = (predictions - m) / s

# m = targets.mean()
# s = np.std(targets)
# targets = (targets - m) / s

## -----------------------------------------------------

# targets = targets[:1500]
# predictions = predictions[:1500]

## -----------------------------------------------------

plt.plot(targets, color=const.colors[0], alpha=1.0, label='targets')
plt.plot(predictions, color=const.colors[1], alpha=0.7, label='prediction')
plt.title(f'{date}/{feature_extractor}/fold_0')
plt.legend()

plt.savefig( pth.join(const.RESULTS_DIR, 'prediction_fold_0.png') )