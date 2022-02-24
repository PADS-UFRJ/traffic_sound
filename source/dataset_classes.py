import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import os
import os.path as pth

import const


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

class FeaturesDataset(Dataset):
    # FramesDataset herda a classe Dataset

    def __init__(self, features_files_list, transform=None):

        ## -----------------------------------------------------
        ## Tratamentos de erro

        for features_file in features_files_list:
            if not pth.isfile(features_file):
                raise Exception('File not found:',features_file)

        ## -----------------------------------------------------
        ## Inicializações

        self.features_files_list = features_files_list # lista com os arquivos npy a serem incluidos incluido no datset

        self.transform = transform # a transformacao a ser aplicada nos vetores de features

        # self.features_shapes = [] # lista com os formatos de cada arquivo npy

        # [NOTE] Estou carregando todos os arrays de features de todos os videos em um unico array na memoria
        # Uma alternativa seria carregar o array de um video para a memoria apenas quando fosse necessario (pode
        # ser mais lento, mas economizaria memoria)

        self.features = []

        for features_file in features_files_list:
            features_array = np.load(features_file)
            # print(f'{features_file}\n  {features_array.shape}')
            for features in features_array:
                self.features.append(features)

        # print(f'features list len: {len(self.features)}')

        self.features = np.array(self.features)

        # print(f'features array shape: {self.features.shape}')


    ################################################################

    def __getitem__(self, index):

        features = self.features[index]

        if self.transform is not None:
            features = self.transform(features)

        return features

    ################################################################

    def __len__(self):

        return len(self.features)


################################################################

## -----------------------------------------------------

# import sys

# sys.path.append(pth.abspath('./'))


# ## -----------------------------------------------------
# # Inicializacoes

# videos_list = const.videos_list

# features_files = [ pth.join(const.FEATURES_DIR, v + '_features.npy')  for v in videos_list ]

# dataset = FeaturesDataset(features_files)

# print(dataset[0].shape)