import os
import numpy as np
import cv2
import const

def stack_images(images_source, save_dir=None):
    """
    Empilhar imagens de um diretorio em um array numpy e salvar em um arquivo npy no diretorio
    correspondente. As imagens nao sao normalizadas. As imagens no diretorio devem ter as mesmas dimensoes.

    (Ver estrutura dos diretorios do dataset no arquivo const.py)
    ----------------
    images_source (str):
        Diretorio contendo as imagens

    save_dir (str):
        O diretorio onde os arquivos npy devem ser salvos. Se nenhum for passado como argumento, os arrays
        sao salvos no diretorio INPUTS_DIR do dataset (ver const.py). Caso INPUTS_DIR nao exista, sera
        criado (bem como seus diretorios pais, se necessario).
    """
    # [TODO] verificar se o diretorio passado possui apenas arquivos validos
    # [TODO] implementar tratamentos de erro
    # [TODO] calcular e retornar/salvar media e desvio padrao (vale a pena?)
    # [TODO] Converter de BRG para RGB (?)
    # [TODO] tratar caso de imagens com numero de canais diferente

    ## -----------------------------------------------------
    ## Tratamentos de erro

    if not os.path.isdir(images_source):
        printc('r','[ERR] ', end='')
        print(f'directory not found: {images_source}')
        exit()

    if (save_dir is not None) and (not os.path.isdir(save_dir)):
        printc('r','[ERR] ', end='')
        print(f'directory not found: {save_dir}')
        exit()

    ## -----------------------------------------------------
    ## Inicalizacoes

    if save_dir is None: # usuario nao forneceu um diretorio
        save_dir = const.INPUTS_DIR # diretorio onde iremos salvar o array npy

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    images_list = os.listdir(images_source) # listar os arquivos de imagem no diretorio
    images_list.sort() # ordenar os nomes dos arquivos

    img_shape = cv2.imread(os.path.join(images_source, images_list[0])).shape

    stacked_images = np.zeros( (len(images_list), 
                                img_shape[0],
                                img_shape[1],
                                3),
                                dtype=np.uint8 ) # array onde as imagens sao empilhadas

    images_source = os.path.normpath(images_source) # removemos a barra '/' do final, caso haja
    stack_filename = os.path.basename(images_source) # pegamos apenas o ultimo diretorio do caminho
    stack_filename += '_stack.npy' # nome do arquivo numpy

    stack_filename = os.path.join(save_dir, stack_filename) # caminho completo do arquivo npy

    ## -----------------------------------------------------
    ## Principal

    for i in range(len(images_list)):
        # [NOTE] O opencv usa o numpy nativamente, mas os arrays sao do formato [B,R,G]
        im_file = os.path.join(images_source, images_list[i])
        stacked_images[i] = cv2.imread(im_file)

    print('reading from: ', images_source)
    print('total images: ', len(images_list))
    print('stack shape: ', stacked_images.shape)
    print('save as: ', stack_filename)

    np.save(stack_filename, stacked_images)
