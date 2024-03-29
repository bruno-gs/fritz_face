# https://github.com/bruno-gs/fritz_face
# USAGE
# python3 mask.py --image images/name_of_image.jpeg

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

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model("mask_detector.model")

	# load the input image from disk, clone it, and grab the image spatial dimensions
	image = cv2.imread(args["image"])
	# orig = image.copy()
	(h, w) = image.shape[:2]

	#function to facilitate image preprocessing for deep learning classification
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
	
	
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# minimum probability to filter weak detections
		if confidence > 0.5:
			
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
						
			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			# For Mask Detection
			faceS = image[startY:endY, startX:endX]
			face = cv2.cvtColor(faceS, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (244, 244))
			face = img_to_array(face)
			face = preprocess_input(face)
			mask_face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face has a mask or not
			(mask, withoutMask) = model.predict(mask_face)[0]

			# determine the class label and color we'll use to draw the bounding box and text
			# Blue -> Mask || Red -> No Mask
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
			# For include the probability in the label
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			size = cv2.getTextSize(label, font, 0.8, 1)[0]
			point = (startX, startY)
			x, y = point
			cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv2.FILLED)
			cv2.putText(image, label, point, font, 0.8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
					
	# Print o total de pessoas reconhecidas
	# cv2.putText(image, "{} pessoa(s)".format(num),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2,cv2.LINE_AA) 
	
	# show the output image
	cv2.imwrite("log/log.jpg", image)


if __name__ == "__main__":
	mask_image()
	im = Image.open("log/log.jpg")
	im.show()
