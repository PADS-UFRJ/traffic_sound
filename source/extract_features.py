import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import torchvision.transforms as transforms

import cv2
import numpy as np

import os
import os.path as pth


# [REF] https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# [REF] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class FramesDataset(Dataset):
    # FramesDataset herda a classe Dataset

    def __init__(self, frames_dirs, transform=None):

        ## -----------------------------------------------------
        ## Tratamentos de erro

        for directory in frames_dirs:
            if not pth.isdir(directory):
                raise Exception('Directory not found:',directory)

        ## -----------------------------------------------------
        ## Inicializações

        self.frames_dirs = frames_dirs # diretorios contendo os frames

        self.transform = transform # a transformacao a ser aplicada nas imagens

        self.frames = [] # lista com o caminho de todos os arquivos de imagem

        for directory in frames_dirs:
            dir_content = os.listdir(directory) # a lista de todos os arquivos presentes no diretorio
            dir_content.sort()

            # [REF] https://docs.python.org/3/library/os.path.html#os.path.splitext
            # [REF] https://www.programiz.com/python-programming/list-comprehension
            # [TODO] tratar outros formatos de imagem
            image_files = [ f for f in dir_content if (pth.splitext(f)[1] == '.png') ]
            image_files = [ pth.join(directory, f) for f in image_files ] # a lista de caminhos completos para os arquivos PNG no diretorio

            self.frames += image_files # adicionamos os arquivos de imagem do diretorio na lista

    ################################################################

    def __getitem__(self, index):

        # img = read_image(self.frames[index])
        img = cv2.imread(self.frames[index])
        # print('cv:',type(img))
        img = transforms.ToPILImage()(img)
        # print('pil:',type(img))

        if self.transform is not None:
            img = self.transform(img)

        return img

    ################################################################

    def __len__(self):

        return len(self.frames)


################################################################


class VggFeatureExtractor(nn.Module):

    def __init__(self, vgg):
        super().__init__() # executamos a inicializacao da classe superior
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        # self.avgpool = vgg.avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
            
    
    def forward(self, x):
        out = self.features(x)
        # print('features:',out.shape)
        out = self.avgpool(out)
        # print('avgpool:',out.shape)
        out = self.maxpool(out)
        # print('maxpool:',out.shape)
        out = self.flatten(out)
        # print('flatten:',out.shape)
    
        return out 


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

MT_DATASET_DIR = '/home/mathlima/dataset'

SOURCE_DIR = pth.dirname( pth.abspath(__file__) )
WORK_DIR = pth.dirname( SOURCE_DIR )

DATASET_DIR = pth.join(WORK_DIR, 'dataset')
RAW_DIR = pth.join(DATASET_DIR, 'raw')
PREPROCESSED_DIR = pth.join(DATASET_DIR, 'preprocessed')
FEATURES_DIR = pth.join(PREPROCESSED_DIR, 'features')
TARGETS_DIR = pth.join(PREPROCESSED_DIR, 'targets')

## -----------------------------------------------------
# Inicializacoes

# videos_list = ['M2U00001MPG']
videos_list = os.listdir( pth.join(MT_DATASET_DIR, 'raw') )
videos_list.sort()
videos_list = [ d.replace('.','') for d in videos_list ] # lista com nomes dos videos

frames_dirs = [ pth.join(MT_DATASET_DIR, v) for v in videos_list ] # lista com os diretorios onde estao os frames de cada video

save_dirs = [ pth.join(FEATURES_DIR, v) for v in videos_list ] # lista com os diretorios onde devemos salvar as features de cada video

# for d in save_dirs:
#     if not pth.isdir(d):
#         os.makedirs(d)

## -----------------------------------------------------
# Inicializacoes

# vgg = models.vgg16
feature_extractor = VggFeatureExtractor( models.vgg16(pretrained=True) )

## -----------------------------------------------------
# Inicializacoes

batch_size = 32

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

## -----------------------------------------------------
# Corpo principal

feature_extractor.to(device)

with torch.no_grad(): # destivamos o calculo de gradiente pois nao estamos treinando o modelo

    for video_index in range(len(videos_list)): # percorremos os videos para extrair as features deles

        if not pth.isdir(save_dirs[video_index]):
            os.makedirs(save_dirs[video_index])

        features_list = []

        dataset = FramesDataset([frames_dirs[video_index]], transform=preprocess)

        dataloader = DataLoader(dataset, batch_size=batch_size)

        print('## -----------------------------------------------------')
        print(f'dataset: {videos_list[video_index]}')
        print(f'  imgs:{len(dataset)}')
        print(f'  batches: {len(dataloader)}')
        print()
        print('Calculating outputs')

        for batch_index in range(len(dataloader)): # o numero de batches que serao executados
            # print(f'- batch {batch_index+1}/{len(dataloader)}')

            imgs_batch = next(iter(dataloader)) # carregamos o proximo batch de imagens de entrada da rede neural

            imgs_batch = imgs_batch.to(device) # carregamos a entrada para a GPU (caso disponivel)

            features = feature_extractor(imgs_batch) # passamos a imagem pela rede neural, obtendo as features

            features = features.cpu() # passamos o tensor para a CPU para podermos salva-lo em disco como array numpy

            fetaures = features.numpy()

            features_list.append(features)

        # features_list[0]: o primeiro batch
        # features_list[0][0]: as features da primeira imagem do primeiro batch
        shape = features_list[0][0].shape # o formato de um vetor de features qualquer (no caso da primeira imagem do primeiro batch)

        shape = (len(dataset),) + shape # o formato final da matriz de features (cada elemento eh um vetor de features de uma imagem)

        features_array = np.empty(shape) # inicalizamos o vetor de features vazio
        # [TODO] verificar se poderiamos usar apenas np.empty(len(dataset))

        print('Stacking all outputs in one array')

        begin = 0
        for i, batch in enumerate(features_list):
            # [NOTE] o ultimo batch possui o mesmo formato dos outros, ainda que haja menos imagens nele
            # por isso usamos o slice no batch para pegar apenas as imagens validas
            if i == len(features_list) - 1:
                end = len(features_array)
            else:
                end = (begin + len(batch))

            features_array[begin:end] = batch[:end-begin]

            begin = end

        filename = f'{videos_list[video_index]}_features.npy'
        filepath = pth.join(FEATURES_DIR, filename)


        np.save(filepath, features_array)

        print(f'Saved in {filepath}')