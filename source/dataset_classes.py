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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

        self.features = np.array(self.features, dtype=np.float64)

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

class AudioTargetDataset(Dataset):
    # herda a classe Dataset

    def __init__(self, files_list, transform=None):

        ## -----------------------------------------------------
        ## Tratamentos de erro

        for f in files_list:
            if not pth.isfile(f):
                raise Exception(f'File not found: {f}')

        ## -----------------------------------------------------
        ## Inicializações

        self.files_list = files_list # files_list[v]: caminho do arquivo npy contendo alvos

        self.transform = transform # a transformacao a ser aplicada nos vetores de alvos

        # [NOTE] Estou carregando todos os arrays em um unico grande array na memoria
        # Uma alternativa seria carregar um array por vez, apenas quando fosse necessario (pode
        # ser mais lento, mas economizaria memoria)

        self.targets = []

        for targets_file in files_list:
            targets_array = np.load(targets_file)
            # print(f'{targets_file}\n  {targets_array.shape}')
            for target in targets_array:
                self.targets.append(target)

        # print(f'targets list len: {len(self.targets)}')

        self.targets = np.array(self.targets)

        # print(f'targets array shape: {self.targets.shape}')


    ################################################################

    def __getitem__(self, index):

        target = self.targets[index].mean()

        if self.transform is not None:
            target = self.transform(target)

        return target

    ################################################################

    def __len__(self):

        return len(self.targets)


################################################################

## -----------------------------------------------------

# import sys

# sys.path.append(pth.abspath('./'))


# ## -----------------------------------------------------
# # Inicializacoes

# videos_list = const.videos_list

# targets_files = [ pth.join(const.MT_DATASET_DIR, v, 'output_targets.npy')  for v in videos_list ]

# dataset = AudioTargetDataset(targets_files)

# print(dataset[0].shape)
# print(dataset[0])