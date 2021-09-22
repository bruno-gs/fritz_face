# AUTOR: FRITZ
# https://github.com/bruno-gs/fritz_face

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
    
    # Carrega o modelo de detecção facial
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join(["face_detector","deploy.prototxt"])
	weightsPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# Carrega o modelo de detecção de máscara
	print("[INFO] loading face mask detector model...")
	model = load_model("mask_detector.model")

	# Carrega a imagem 
	image = cv2.imread(args["image"])
	# Obtem as dimensões esaciais da imagem
    (h, w) = image.shape[:2]

	# Facilitar o pré-processamento de imagem para classificação de deep learning
	blob = cv2.dnn.blobFromImage(image, 1.0, (700,700), (104.0, 177.0, 123.0))
	
	# Passa o blob pela rede e pega as detecções de rosto
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

    for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# probabilidade para as detecções
		if confidence > 0.5:
            





# chamando a função principal e iniciando o código
if __name__ == "__main__":

    # função principal
	mask_age_gender()

    # salva a imagem gerada em um pasta log
    # conferencia posterior ou  caso feche a janela de resultado
	im = Image.open("log/log.jpg")

    # janela de resultado
	im.show()