from os import path

WORK_DIR = path.dirname( path.abspath(__file__) )

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