# AUTOR: FRITZ
# https://github.com/bruno-gs/fritz_recognizer_image

###################################################
# COMO USAR
#
# python3 full.py --image images/name_of_image.jpeg
# também aceita .jpg

#############################################################################
# pacotes necessários
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
from PIL import Image

##############################################################################


def mask_age_gender():

    # construção dos argumentos de inicialização
    arg = argparse.ArgumentParser()
	arg.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(arg.parse_args())


# chamando a função principal e iniciando o código
if __name__ == "__main__":

    # função principal
	mask_age_gender()

    # salva a imagem gerada em um pasta log
    # conferencia posterior ou  caso feche a janela de resultado
	im = Image.open("log/log.jpg")

    # janela de resultado
	im.show()