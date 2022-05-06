
import os
import os.path as pth

MT_DATASET_DIR = '/home/mathlima/dataset' # diretorio do dataset na pasta pessoal do Matheus

SOURCE_DIR = pth.dirname( pth.abspath(__file__) )

WORK_DIR = pth.dirname( SOURCE_DIR )

RESULTS_DIR = pth.join(WORK_DIR, 'results')

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

###############################################################################

# Cores bonitas para serem usadas no grafico
# [REF] https://towardsdatascience.com/making-matplotlib-beautiful-by-default-d0d41e3534fd

CB91_Blue   = '#2CBDFE'
CB91_Green  = '#47DBCD'
CB91_Pink   = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber  = '#F5B14C'

colors = [CB91_Amber,
            CB91_Purple,
            CB91_Green,
            CB91_Pink,
            CB91_Blue,
            CB91_Violet]