import torch
from torch import nn
import torchvision as vision
import json


def reset_parameters(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer: {layer}')
            layer.reset_parameters()
        else:
            # print(f"Nao possui reset_parameters: {layer}")
            for sublayer in layer.children():
                if hasattr(sublayer, 'reset_parameters'):
                    # print(f'Reset trainable parameters of layer: {sublayer}')
                    sublayer.reset_parameters()
                # else:
                #     print(f"Nao possui reset_parameters: {sublayer}")
    # for child in m.children():
    #     for layer in child.children():
    #         if hasattr(layer, 'reset_parameters'):
    #             print(f'Reset trainable parameters of layer: {layer}')
    #             layer.reset_parameters()
    #         else:
    #             print(f"Nao possui reset_parameters: {layer}")


class VggFeatureExtractor(nn.Module):

    def __init__(self, vgg):
        super().__init__() # executamos a inicializacao da classe superior
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
            
    def reset_parameters(self):
        reset_parameters(self)
    
    def forward(self, x):
        x = self.features(x)
        # print('features:', x.shape)

        x = self.avgpool(x)
        # print('avgpool:', x.shape)

        # x = self.maxpool(x)
        # print('maxpool:', x.shape)

        x = self.flatten(x)
        # print('flatten:', x.shape)
    
        return x 


# Arquitetura da rede FC
class FCNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers_size_list, dropout_value):
        super().__init__()

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

        self.double()

    def reset_parameters(self):
        reset_parameters(self)

    def forward(self, x):
        for layer in self.layers[:-1]: # Estou pegando todas as camadas,exceto a última 
            x = torch.tanh(layer(x))
        x = self.dropout(x)
        x = self.layers[-1](x)
        return x     