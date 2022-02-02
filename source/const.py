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