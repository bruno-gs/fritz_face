# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

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
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load the input image from disk, clone it, and grab the image spatial dimensions
	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# (700,700) para imagens de 1280x720 - otimo
	#function to facilitate image preprocessing for deep learning classification
	blob = cv2.dnn.blobFromImage(image, 1.0, (700,700), (104.0, 177.0, 123.0))
	
	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()
	num = 0	
	for i in detections:
		for j in i:
			for k in j:				
				if k[2] > 0.5:
					num+=1 #print(num) quantidade de pessoas identificadas	
	
	weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
	model_name, img_size = Path(weight_file).stem.split("_")[:2]
	img_size = int(img_size)
	cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
	model1 = get_model(cfg)
	model1.load_weights(weight_file)
	
	i=0
	j=0
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			#print(confidence)
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			faces = np.empty((1, img_size, img_size, 3))
			x1, y1, x2, y2, w1, h1 = startX , startY , endX + 1, endY + 1, w, h
			xw1 = max(int(x1 - 0 * w1), 0)
			yw1 = max(int(y1 - 0 * h1), 0)
			xw2 = min(int(x2 + 0 * w1), w - 1)
			yw2 = min(int(y2 + 0 * h1), h - 1)
			faces[0] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
			
			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			faceS = image[startY:endY, startX:endX]
			face = cv2.cvtColor(faceS, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (244, 244))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			

			# include the probability in the label
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			if label == "Mask":
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				size = cv2.getTextSize(label, font, 0.8, 1)[0]
				point = (startX, startY)
				x, y = point
				cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv2.FILLED)
				cv2.putText(image, label, point, font, 0.8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
			else:
				results = model1.predict(faces)
				predicted_genders = results[0]
				#print(predicted_genders)
				ages = np.arange(0, 101).reshape(101, 1)
				predicted_ages = results[1].dot(ages).flatten()
				#print(predicted_ages)
				
				label1 = "{}, {}".format(int(predicted_ages[0]), "M" if predicted_genders[0][0] < 0.5 else "F")
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				size = cv2.getTextSize(label, font, 0.8, 1)[0]
				size1 = cv2.getTextSize(label1, font, 0.8, 1)[0]
				point = (startX, startY)
				x, y = point
				point1 = (startX,endY)
				x1, y1 = point1
				cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv2.FILLED)
				cv2.putText(image, label, point, font, 0.8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
				cv2.rectangle(image, (x1, y1 - size1[1]), (x1 + size1[0], y1), color, cv2.FILLED)
				cv2.putText(image, label1, point1, font, 0.8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
			i+=1
			#draw
					
	
	cv2.putText(image, "{} pessoa(s)".format(num),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2,cv2.LINE_AA) #total de pessoas
	# show the output image
	cv2.imwrite("log/log.jpg", image)
	#cv2.imshow("Output", image)   # Aparecer imagem na hora
	#cv2_plt_imshow(image)

if __name__ == "__main__":
	mask_image()
	im = Image.open("log/log.jpg")
	im.show()
