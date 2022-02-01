import const
from os import path

def extract_from_video(video_file, downsampling_factor=30):
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

    extract_images_from_video(video_file, downsampling_factor)
    extract_audio_from_video(video_file, downsampling_factor)



def extract_images_from_video(video_file, downsampling_factor=30)
    """
    Extrair frames de um video e salvar no diretorio correspondente do dataset.

    (Ver estrutura dos diretorios no arquivo folder_structure)
    ----------------
    video_file (str):
        arquivo do video fonte

    downsampling_factor (int):
        fator de reamostragem. Isto eh, se downsampling_factor == X, pegamos 1 frame de video a
        cada X frames.
    """

    return -1

def extract_audio_from_video(video_file, downsampling_factor=30)
    """
    Extrair array numpy representando audio de um video e salvar no diretorio correspondente do
    dataset. Mais especificamente, armazenamos o log da potencia do som ao longo de um intervalo
    determinado pelo fator de reamostragem.

    (Ver estrutura dos diretorios no arquivo folder_structure)
    ----------------
    video_file (str):
        arquivo do video fonte

    downsampling_factor (int):
        fator de reamostragem. Isto eh, se downsampling_factor == X, pegamos 1 frame de video a
        cada X frames. O numero final de amostras de audio sera igual ao numero de frames que
        pegamos.
    """

    return -1