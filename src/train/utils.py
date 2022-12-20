
# Listas necessárias para a busca por parametros
epochs = [100]
opt = ['adam']
batch = [32]
dropout = [0.5] # dropout 0 não dá certo !
dropout_lstm = [0.2]
lr = [1e-4]

SCHEDULER = True
NUMBER_STEPS_EPOCHS = 40
lr_scheduler = [5e-6,1e-6,5e-7]

# Constantes 
LSTM = True
FEATURES = 'Felipe' # 'Felipe' ou 'Matheus' ou 'torch_model_with_weights_of_tf/keras' 
size_windows = 32 
num_layers = 1 # Número de camadas lstm empilhadas no modelo
bidirectional = False
option_overlap = True
option_causal = False

OPTION_SHUFFLE = True
OPTION_NUM_WORKERS = 3
INPUT_SIZE_FEATURES = 512
HIDDEN_SIZE = 128
OUTPUT_SIZE_FEATURES = 1
SEED_NUMBER = 22

PATH_DATA_TO_EXTRACTION = '/home/caroline/traffic-analysis/dataset/' # Caminho dos dados usados para a extração
PATH_EXTRACTED_FEATURES = '/home/caroline/Traffic-Pytorch/Data/Preprocessed/' # Caminho onde salvo as features e os targets  '
EXTRACTION_MODEL = 'vgg16'
PATH_FEATURES_CAROL = PATH_EXTRACTED_FEATURES+EXTRACTION_MODEL+'/'
VIDEOS_NUMBER = 38
PATH_FEATURES_FELIPE = '/home/felipevr/traffic_sound/dataset/preprocessed/features/vgg16/'
PATH_TARGETS_FELIPE = '/home/felipevr/traffic_sound/dataset/preprocessed/targets/'
PATH_FEATURES_PYTORCH_MODEL_TF_KERAS_WEIGHTS = '/home/caroline/traffic_sound/src/extraction/traffic_sound/dataset/preprocessed/features_torch/'

# Lista de vídeos
videos_list = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00017MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00041MPG",
    "M2U00042MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG"
]