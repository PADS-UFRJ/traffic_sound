from os import path

WORK_DIR = path.dirname( path.split(__file__)[0] )

SOURCE_DIR = path.join(WORK_DIR, 'source')
RESULTS_DIR = path.join(WORK_DIR, 'results')
DATASET_DIR = path.join(WORK_DIR, 'dataset')

RAW_DIR = path.join(DATASET_DIR, 'raw')
PREPROCESSED_DIR = path.join(DATASET_DIR, 'preprocessed')

VIDEOS_DIR = path.join(RAW_DIR, 'videos')
AUDIO_DIR = path.join(RAW_DIR, 'audio')
IMAGES_DIR = path.join(RAW_DIR, 'images')

INPUTS_DIR = path.join(PREPROCESSED_DIR, 'inputs')
TARGETS_DIR = path.join(PREPROCESSED_DIR, 'targets')
FEATURES_DIR = path.join(PREPROCESSED_DIR, 'features')

videos_list = ['M2U00001.MPG',
                'M2U00002.MPG']

"""
    ** ESTRUTURA DOS DIRETORIOS **
    Em algumas funcoes, eh possivel alterar ou especificar o diretorio de destino/origem pelos argumentos,
    mas sempre que forem usados os diretorios padrao, a estrutura considerada eh a dada abaixo.
    Alternativamente, o usuario pode escrever novas constantes, com os diretorios desejados, similar

    . WORK_DIR
    |-- SOURCE_DIR
    |   |-- const.py
    |   |-- util.py
    |   '-- ...
    |-- DATASET_DIR
    |   |-- RAW_DIR
    |   |   |-- VIDEOS_DIR
    |   |   |   |-- video01.mp4
    |   |   |   '-- video02.avi
    |   |   |-- IMAGES_DIR
    |   |   |   |-- video01mp4
    |   |   |   |   |-- video01mp4_frame01.png
    |   |   |   |   |-- video01mp4_frame02.png
    |   |   |   |   |-- video01mp4_frame03.png
    |   |   |   |   '-- ...
    |   |   |   '-- video02avi
    |   |   |       |-- video02avi_frame01.png
    |   |   |       |-- video02avi_frame02.png
    |   |   |       '-- ...
    |   |   '-- AUDIO_DIR
    |   |       |-- video01mp4_audio.wav
    |   |       '-- video02avi_audio.wav
    |   '-- PREPOCESSED_DIR
    |       |-- INPUTS_DIR
    |       |   |-- video01mp4_inputs.npy
    |       |   |-- video02avi_inputs.npy
    |       |   |-- fold_0_trn_inputs.npy
    |       |   |-- fold_0_val_inputs.npy
    |       |   '-- ...
    |       |-- TARGETS_DIR
    |       |   |-- video01mp4_targets.npy
    |       |   |-- video02avi_targets.npy
    |       |   |-- fold_0_trn_targets.npy
    |       |   |-- fold_0_val_targets.npy
    |       |   '-- ...
    |       '-- FEATURES_DIR
    |           |-- video01mp4_features.npy
    |           |-- video02avi_features.npy
    |           |-- fold_0_trn_features.npy
    |           |-- fold_0_val_features.npy
    |           '-- ...
    '-- RESULTS_DIR

"""