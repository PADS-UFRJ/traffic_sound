import const
import cv2
import math
import os
from utils import printc
from shutil import which

def extract_from_video(video_file, downsampling_factor=30, image_shape=(224,224)):
    """
    Extrair frames de um video e array numpy representando o audio e salva-los no diretorio correspondente
    do dataset. 
    
    (Para mais detalhes ver extract_images_from_video e extract_audio_from_video)
    (Ver estrutura dos diretorios no arquivo folder_structure)
    ----------------
    video_file (str):
        arquivo do video fonte

    downsampling_factor (int):
        fator de reamostragem. Isto eh, se downsampling_factor == X, pegamos 1 frame de video a
        cada X frames. O numero final de amostras de audio sera igual ao numero de frames que
        pegamos.
    """
    # [TODO] atualizar descricao e argumentos para incluir argumentos de extract_images_from_video

    extract_images_from_video(video_file, downsampling_factor)
    extract_audio_from_video(video_file, downsampling_factor)

###############################################################


def extract_images_from_video(video_file, downsampling_factor=30, image_shape=(224,224), save_dir=None):
    """
    Extrair frames de um video e salva-los como arquivos de imagem no diretorio correspondente.

    (Ver estrutura dos diretorios do dataset no arquivo const.py)
    ----------------
    video_file (str):
        Arquivo do video fonte

    downsampling_factor (int):
        Fator de reamostragem. Isto eh, se downsampling_factor == X, pegamos 1 frame de video a
        cada X frames.

    image_shape (int list/tuple):
        As dimensoes (em pixels) com as quais salvar as imagens. Deve ser uma lista ou tupla de inteiros
        positivos, no formato [largura, altura].

    save_dir (str):
        O diretorio onde as imagens devem ser salvas. Se nenhum for passado como argumento, as imagens
        sao salvas no diretorio IMAGES_DIR do dataset (ver const.py). Caso IMAGES_DIR nao exista, sera
        criado (bem como seus diretorios pais, se necessario).
    """

    # [TODO] Retirar/melhorar prints
    # [TODO] Substituir exit() por raise() nos tratamentos de erro
    # [TODO] Implementar verificacoes do tipo dos argumentos passados

    ## -----------------------------------------------------
    ## Tratamentos de erro

    if not os.path.isfile(video_file):
        printc('r','[ERR] ', end='')
        print(f'file not found: {video_file}')
        exit()

    if (save_dir is not None) and (not os.path.isdir(save_dir)):
        printc('r','[ERR] ', end='')
        print(f'directory not found: {save_dir}')
        exit()

    if downsampling_factor <= 0:
        printc('r','[ERR] ', end='')
        print(f'downsampling_factor must be a a positive integer, not: ({type(downsampling_factor)}) {downsampling_factor}')
        exit()

    if (image_shape[0] <= 0) or (image_shape[1] <= 0):
        printc('r','[ERR] ', end='')
        print(f'image_shape elements must be positive integers, not: {image_shape}')
        exit()

    ## -----------------------------------------------------
    ## Inicalizacoes

    video_name = os.path.split(video_file)[1] # pegando apenas o nome do arquivo, caso tenha sido fornecido o caminho completo
    video_name = video_name.replace(' ', '') # formatando a string para retirar espacos em branco
    video_name = video_name.replace('.', '') # formatando a string para retirar o ponto da extensao

    
    video_cap = cv2.VideoCapture(video_file)

    video_total_frames = int( video_cap.get(cv2.CAP_PROP_FRAME_COUNT) ) # obtendo o total de frames do video a partir da propriedade CAP_PROP_FRAME_COUNT
    downsample_total_frames = int( video_total_frames / downsampling_factor ) # obtendo o total de frames apos reamostragem

    downsample_total_digits = int(math.log10( downsample_total_frames )) + 1 # obtendo o numero de digitos que precisamos para escrever todos os indices de frames apos a reamostragem

    video_frame_index = 0 # indice do frame original do video
    downsample_frame_index = 0 # indice do frame ja reamostrado

    if save_dir is None: # usuario nao forneceu um diretorio
        save_dir = os.path.join(const.IMAGES_DIR, video_name) # diretorio onde iremos salvar os frames

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)


    print('video_name:',video_name)
    print('video_fps:',video_cap.get(cv2.CAP_PROP_FPS))
    print('video_total_frames:',video_total_frames)
    print('downsample_total_frames:',downsample_total_frames)
    print('downsample_total_digits:',downsample_total_digits)
    print('save_dir:',save_dir)

    ## -----------------------------------------------------
    ## Principal

    ret, frame = video_cap.read()
    while ret:

        if (video_frame_index % downsampling_factor == 0):
            print(downsample_frame_index)
            # frame_name eh o nome do arquivo de imagem em que salvamos o frame (ex: '~/data/img_dir/vid1/vid1_frame0013.png')
            # [REF] Martin Pieters em https://stackoverflow.com/questions/18004646/dynamically-calculated-zero-padding-in-format-string-in-python
            frame_name = video_name + '_frame{number:0{padding}d}.png'.format(number=downsample_frame_index,
                                                                            padding=downsample_total_digits)
            frame_name = os.path.join(save_dir, frame_name)

            if image_shape is not None:
                frame = cv2.resize(frame, (image_shape[0], image_shape[1]))

            cv2.imwrite(frame_name, frame) # salvar o frame em disco

            downsample_frame_index += 1

        ret, frame = video_cap.read() # ler o proximo frame (se houver)
        video_frame_index += 1

    video_cap.release() # ao final, liberar a captura do video

###############################################################

def extract_audio_from_video(video_file, save_dir=None):
    """
    Extrair audio de um video e salvar em arquivo wav no diretorio correspondente.

    (Ver estrutura dos diretorios do dataset no arquivo const.py)
    ----------------
    video_file (str):
        Arquivo do video fonte

    save_dir (str):
        O diretorio onde os arquivos devem ser salvos. Se nenhum for passado como argumento, os audios
        sao salvos no diretorio AUDIO_DIR do dataset (ver const.py). Caso AUDIO_DIR nao exista, sera
        criado (bem como seus diretorios pais, se necessario).
    """

    # [TODO] Colocar/melhorar prints
    # [TODO] Substituir exit() por raise() nos tratamentos de erro
    # [TODO] Implementar verificacoes do tipo dos argumentos passados

    ## -----------------------------------------------------
    ## Tratamentos de erro

    if not os.path.isfile(video_file):
        printc('r','[ERR] ', end='')
        print(f'file not found: {video_file}')
        exit()

    if (save_dir is not None) and (not os.path.isdir(save_dir)):
        printc('r','[ERR] ', end='')
        print(f'directory not found: {save_dir}')
        exit()

    if which('ffmpeg') is None:
        printc('r','[ERR] ', end='')
        print(f'ffmpeg is not installed. Cant extract audio.')
        exit()

    ## -----------------------------------------------------
    ## Inicalizacoes

    video_name = os.path.split(video_file)[1] # pegando apenas o nome do arquivo, caso tenha sido fornecido o caminho completo
    video_name = video_name.replace(' ', '') # formatando a string para retirar espacos em branco
    video_name = video_name.replace('.', '') # formatando a string para retirar o ponto da extensao

    if save_dir is None: # usuario nao forneceu um diretorio
        save_dir = const.AUDIO_DIR # diretorio onde iremos salvar o arquivo de audio

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    audio_name = video_name + '_audio.wav'
    audio_name = os.path.join(save_dir, audio_name)

    ## -----------------------------------------------------
    ## Principal

    # Daremos um comando usando o programa ffmpeg para extrair o audio do video
    os_command = "ffmpeg -loglevel quiet -i " + video_file + " " + audio_name
    os.system(os_command)