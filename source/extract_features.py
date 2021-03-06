import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision as vision
import torchvision.transforms as transforms

import cv2
import numpy as np

import os
import os.path as pth

import const
from dataset_classes import FramesDataset
from models import VggFeatureExtractor


################################################################
# MAIN

preprocess = transforms.Compose([
                                transforms.Resize( (224,224) ),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## -----------------------------------------------------

import sys

sys.path.append(pth.abspath('./'))

## -----------------------------------------------------
# Declaracao de constantes

videos_list = const.videos_list

frames_dirs = [ pth.join(const.MT_DATASET_DIR, v) for v in videos_list ] # lista com os diretorios onde estao os frames de cada video

save_dirs = [ pth.join(const.FEATURES_DIR, v) for v in videos_list ] # lista com os diretorios onde devemos salvar as features de cada video

# for d in save_dirs:
#     if not pth.isdir(d):
#         os.makedirs(d)

## -----------------------------------------------------
# Inicializacoes

# vgg = models.vgg16
feature_extractor = VggFeatureExtractor( vision.models.vgg16(pretrained=True) )

## -----------------------------------------------------
# Inicializacoes

batch_size = 32

gpu = '1'

if not torch.cuda.is_available():
    raise Exception('GPU not available.')

device = torch.device('cuda:' + gpu)

## -----------------------------------------------------
# Corpo principal

feature_extractor.to(device)

with torch.no_grad(): # destivamos o calculo de gradiente pois nao estamos treinando o modelo

    for video_index in range(len(videos_list)): # percorremos os videos para extrair as features deles

        if not pth.isdir(save_dirs[video_index]):
            os.makedirs(save_dirs[video_index])

        features_list = []

        dataset = FramesDataset([frames_dirs[video_index]], transform=preprocess)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print('## -----------------------------------------------------')
        print(f'dataset: {videos_list[video_index]}')
        print(f'  imgs:{len(dataset)}')
        print(f'  batches: {len(dataloader)}')
        print()
        print('Getting batches and calculating their outputs')

        for batch_index, imgs_batch in enumerate(dataloader): # percorrendo os batches de imagens

            # print(f'- batch {batch_index+1}/{len(dataloader)}')
            # print(f'--- {imgs_batch.shape} frames')

            imgs_batch = imgs_batch.to(device) # carregamos a entrada para a GPU (caso disponivel)

            features = feature_extractor(imgs_batch) # passamos a imagem pela rede neural, obtendo as features

            features = features.cpu() # passamos o tensor para a CPU para podermos salva-lo em disco como array numpy

            features = features.numpy()

            for feat in features:
                features_list.append(feat)

        print('Stacking all outputs in one array')

        features_array = np.array(features_list)

        print(f'  shape: {features_array.shape}')

        filename = f'{videos_list[video_index]}_features.npy'
        filepath = pth.join(const.FEATURES_DIR, filename)

        np.save(filepath, features_array)

        print(f'Saved in {filepath}')