# https://github.com/bruno-gs/fritz_face
# USAGE
# python3 age_gender.py --image images/name_of_image.jpeg

# import the necessary packages
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

# packages out -- Conditions
# import face_recognition
# import dlib
# from contextlib import contextmanager
# from PIL import ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# from cv2_plt_imshow import cv2_plt_imshow, plt_format

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join(["face_detector","deploy.prototxt"])
	weightsPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)
	
	# load the input image from disk, clone it, and grab the image spatial dimensions
	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# (700,700) para imagens de 1280x720 - otimo
	blob = cv2.dnn.blobFromImage(image, 1.0, (700,700), (104.0, 177.0, 123.0))
	
	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	# For the total of recognized faces
	num = 0	
	for i in detections:
		for j in i:
			for k in j:				
				if k[2] > 0.5:
					num+=1 	
	
	weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
	model_name, img_size = Path(weight_file).stem.split("_")[:2]
	img_size = int(img_size)
	cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
	model_age = get_model(cfg)
	model_age.load_weights(weight_file)

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# minimum probability to filter weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			age_faces = np.empty((1, img_size, img_size, 3))
			x1, y1, x2, y2, w1, h1 = startX , startY , endX + 1, endY + 1, w, h
			xw1 = max(int(x1 - 0 * w1), 0)
			yw1 = max(int(y1 - 0 * h1), 0)
			xw2 = min(int(x2 + 0 * w1), w - 1)
			yw2 = min(int(y2 + 0 * h1), h - 1)
			age_faces[0] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			color = (0, 0, 255)
			results = model_age.predict(age_faces)
			predicted_genders = results[0]
			ages = np.arange(0, 101).reshape(101, 1)
			predicted_ages = results[1].dot(ages).flatten()	
			label1 = "{}, {}".format(int(predicted_ages[0]), "M" if predicted_genders[0][0] < 0.5 else "F")
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			size = cv2.getTextSize(label1, font, 0.8, 1)[0]
			point1 = (startX,endY)
			x1, y1 = point1
			cv2.rectangle(image, (x1, y1 - size[1]), (x1 + size[0], y1), color, cv2.FILLED)
			cv2.putText(image, label1, point1, font, 0.8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
			
	cv2.imwrite("log/log.jpg", image)
	
if __name__ == "__main__":
	mask_image()
	im = Image.open("log/log.jpg")
	im.show()

