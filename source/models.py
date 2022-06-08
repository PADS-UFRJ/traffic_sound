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

    def __init__(self, vgg, name=None):
        super().__init__() # executamos a inicializacao da classe superior

        if vgg == 'vgg16':
            self.name = 'vgg16_feat_xtr' if name is None else name
            vgg = vision.models.vgg16(pretrained=True)
        else:
            if vgg == 'vggNT100':
                self.name = 'vggNT100_feat_xtr' if name is None else name
                pretrained_model_file = '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg/barlowtwins_vgg16.pth'
            elif vgg == 'vggNT500':
                self.name = 'vggNT500_feat_xtr' if name is None else name
                pretrained_model_file = '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_e500_2/barlowtwins_vgg16.pth'
            elif vgg == 'vggNT3k':
                self.name = 'vggNT3k_feat_xtr' if name is None else name
                pretrained_model_file = '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_queue_192_e3000_20220511/barlowtwins_vgg16.pth'
            else:
                raise Exception(f'Unknown model chosen: {vgg}')

            # inicializando modelo
            vgg = vision.models.vgg16(pretrained=False)

            # carregando pesos do disco
            state_dict = torch.load(pretrained_model_file, map_location='cpu')

            # inicializando os pesos do modelo com os pesos carregados do disco
            missing_keys, unexpected_keys = vgg.load_state_dict(state_dict, strict=False)

            expected_missing_keys = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
            assert (missing_keys == expected_missing_keys) and (unexpected_keys == [])

        
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=7,stride=7,padding=0,dilation=1, ceil_mode = False)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

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

    def __init__(self, input_size, output_size, hidden_layers_size_list, dropout_value=None, name=None):
        super().__init__()

        self.name = 'FCNetwork' if name is None else name

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_size_list = hidden_layers_size_list
        size_current = input_size
        self.layers = nn.ModuleList()
        for size_index in hidden_layers_size_list:
            self.layers.append(nn.Linear(size_current, size_index))
            size_current = size_index
        self.layers.append(nn.Linear(size_current, output_size))
        if dropout_value is not None:
            if dropout_value <= 0:
                raise Exception(f'Non-positive value of dropout given')
            self.dropout = nn.Dropout(dropout_value)
        else:
            self.dropout = None

        self.double()

    def reset_parameters(self):
        reset_parameters(self)

    def forward(self, x):
        for layer in self.layers[:-1]: # Estou pegando todas as camadas,exceto a última 
            x = torch.tanh(layer(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
