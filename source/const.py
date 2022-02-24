
import os
import os.path as pth

MT_DATASET_DIR = '/home/mathlima/dataset' # diretorio do dataset na pasta pessoal do Matheus

SOURCE_DIR = pth.dirname( pth.abspath(__file__) )

WORK_DIR = pth.dirname( SOURCE_DIR )

DATASET_DIR = pth.join(WORK_DIR, 'dataset')
RAW_DIR = pth.join(DATASET_DIR, 'raw')
PREPROCESSED_DIR = pth.join(DATASET_DIR, 'preprocessed')
FEATURES_DIR = pth.join(PREPROCESSED_DIR, 'features')
TARGETS_DIR = pth.join(PREPROCESSED_DIR, 'targets')

## -----------------------------------------------------

# videos_list = ['M2U00001MPG']
videos_list = os.listdir( pth.join(MT_DATASET_DIR, 'raw') )
videos_list.sort()
videos_list = [ d.replace('.','') for d in videos_list ] # lista com nomes dos videos
