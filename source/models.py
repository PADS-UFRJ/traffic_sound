import time
import torch
from torch import nn
import torchvision as vision
from torch.autograd import Variable 
import json


def recursive_reset_parameters(m):
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
            recursive_reset_parameters(layer)


class VggFeatureExtractor(nn.Module):

    def __init__(self, vgg, name=None):
        super().__init__() # executamos a inicializacao da classe superior

        self.name = (str(vgg) + "_feat_extr") if name is None else name
        if vgg == 'vgg16':
            vgg = vision.models.vgg16(pretrained=True)
        else:
            known_models = {
                'vggNT100':          '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg/barlowtwins_vgg16.pth',
                'vggNT500':          '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_e500_2/barlowtwins_vgg16.pth',
                'vggNT3k':           '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_queue_192_e3000_20220511/barlowtwins_vgg16.pth',
                'vggNT6k_subset50':  '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_queue_192_e6000_subset_50_20220609/barlowtwins_vgg16.pth',
                'vggNT12k_subset25': '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_batch_64_p4096_vgg_queue_192_e12000_subset_25_20220627/barlowtwins_vgg16.pth',
                'vggCOR3k':          '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_corrio_batch_64_p4096_vgg_queue_192_e3000_20220704/barlowtwins_vgg16.pth',
                'vggCOR6k_subset50': '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_corrio_batch_64_p4096_vgg_queue_192_e6000_subset_50_20230405/barlowtwins_vgg16.pth',
                'vggCOR12k_subset25': '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_corrio_batch_64_p4096_vgg_queue_192_e12000_subset_25_20230421/barlowtwins_vgg16.pth',
                'vggCOR24k_subset12_5': '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_corrio_batch_64_p4096_vgg_queue_192_e24000_subset_12_5_20230617/barlowtwins_vgg16.pth',
                'vggNTCOR3k':        '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_corrio_batch_64_p4096_vgg_queue_192_e3000_20220816/barlowtwins_vgg16.pth',
                'vggNTCOR1k5':       '/home/pedrocayres/unsupervised/barlowtwins/checkpoint_nittrans_corrio_batch_64_p4096_vgg_queue_192_e1500_20220826/barlowtwins_vgg16.pth'
            }

            if vgg not in known_models:
                raise Exception(f'Unknown model chosen: {vgg}')

            pretrained_model_file = known_models[vgg]

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
        # [TODO] recarregar pesos de arquivo ao inves de reiniciar todos os pesos
        recursive_reset_parameters(self)
    
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
        recursive_reset_parameters(self)

    def forward(self, x):
        for layer in self.layers[:-1]: # Estou pegando todas as camadas,exceto a última 
            x = torch.tanh(layer(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


class ModelFromDict(nn.Module):
    # espera-se que o dicionario seja fruto da leitura de um arquivo JSON algumas chaves sao
    # inseridas no dicionario antes da declaracao do modelo de acordo com os parametros de treino
    # p.ex.: dropout da camada FC

    def __init__(self, config_dict):
        super().__init__()

        self.name = config_dict['name']

        layers_dict = config_dict['layers']

        layers = []

        # percorremos os itens do dicionario e, para cada um, adicionamos a camada correspondente
        for layer_key in layers_dict:
            layer = layers_dict[layer_key]

            if (layer_key == 'FeatureExtractor') or (layer_key == 'feature_extractor'):
                if layer['offline'] == False:
                    layers.append( VggFeatureExtractor( layer['model'] ) )

            if (layer_key == 'FCNetwork') or (layer_key == 'fc_network'):
                layers.append(
                                FCNetwork(
                                        input_size=layer['input_size'],
                                        hidden_layers_size_list=layer['hidden_layers'],
                                        output_size=layer['output_size'],
                                        dropout_value=layer['dropout_value']
                                )
                            )
            
            if (layer_key == 'LSTMNetwork') or (layer_key == 'lstm_network'):
                layers.append(
                                LSTMNetwork(
                                        input_size=layer['input_size'],
                                        hidden_size=layer['hidden_size'],
                                        hidden_dropout_value=layer['fc_dropout'],
                                        lstm_num_layers=layer['lstm_num_layers'],
                                        lstm_dropout_value=layer['lstm_dropout'],
                                        output_size=layer['output_size'],
                                        bidirectional=layer['bidirectional'],
                                        device=layer['device'],
                                )
                            )

        self.layers = nn.Sequential(*layers)

    def reset_parameters(self):
        recursive_reset_parameters(self)

    def forward(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        return self.layers(x)

class LSTMNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size: int, hidden_dropout_value: float, lstm_num_layers: int, lstm_dropout_value: float, bidirectional: bool, device: torch.device):
        super().__init__() 

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_dropout_value = hidden_dropout_value
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout_value = lstm_dropout_value
        self.bidirectional = bidirectional
        self.num_directions = self.bidirectional + 1
        self.device = device

        self.dropout_lstm_layer = nn.Dropout(self.lstm_dropout_value)
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.lstm_num_layers,
                            batch_first=True,
                            dropout=self.lstm_dropout_value,
                            bidirectional=self.bidirectional)
        self.dense_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout_linear_layer = nn.Dropout(self.hidden_dropout_value)
        self.dense = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.dropout_lstm_layer(x)

        # Inicializando os estados ocultos com tensores preenchidos por 0  
        h_0 = Variable(torch.zeros(self.lstm_num_layers*self.num_directions, x.size(0), self.hidden_size)) # hidden state
        c_0 = Variable(torch.zeros(self.lstm_num_layers*self.num_directions, x.size(0), self.hidden_size)) # internal state

        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)

        # Alimentando a camada lstm com os estados ocultos e com os dados.
        #  
        # output = contém todos os estados ocultos ao longo da sequência.
        # (h_n, c_n) = tensor com os últimos estados da camada lstm

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0)) # shape de h_n -> [1,32,128]

        # Remodelando os dados para poder usar na camada dense 
        h_n = h_n.view(-1, self.hidden_size) # shape de h_n -> [32,128]

        x = torch.tanh(self.dense_hidden(h_n))
        x = self.dropout_linear_layer(x)
        x = self.dense(x)
        return x 