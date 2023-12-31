# RUNS WITH TENSOR ENVIRONMENT

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from fileinput import filename
import cv2
import os
import tensorflow as tf
import numpy as np
import sys
#import depthai as dai 

FRAME_SHAPE = 256

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

new_model = tf.keras.models.load_model(resource_path(r"model_7_paintline"))
new_model2 = tf.keras.models.load_model(resource_path(r"resnet_paint_2"))

def load_and_prep_image(filename,img_shape=FRAME_SHAPE):
	"""
	Reads an image from filename, turns it into a tensor and reshapes it to the selected shape (eg 224)
	"""
	#read in the image (modified because opencv)
	#img = tf.io.read_file(filename)
	img = filename
	#decode the image into a tensor
	img = tf.image.decode_image(img)
	#resize the image
	img = tf.image.resize(img, size=[img_shape,img_shape])
	#rescale the image
	img = img/255
	return img

font = cv2.FONT_HERSHEY_SIMPLEX

#file_list = os.listdir(resource_path(r"resources\mix"))
file_list = os.listdir(r"E:\temp")
n=0
for filepaths in file_list:
	#ruta_c = resource_path(r"resources\mix")+"\\"+filepaths
	ruta_c = r"E:\temp"+"\\"+filepaths
	img = cv2.imread(ruta_c)
	"""
	y=0
	x=0
	h=720 #1080
	w=920 #1350
	crop = img[y:y+h, x:x+w]
	"""
	image = cv2.resize(img,dsize=(224,224), interpolation = cv2.INTER_CUBIC)
	image2 = cv2.resize(img,dsize=(FRAME_SHAPE,FRAME_SHAPE), interpolation = cv2.INTER_CUBIC)
	final_data = new_model.predict(np.expand_dims(image, axis=0),verbose=0)
	final_data2 = new_model2.predict(np.expand_dims(image2, axis=0),verbose=0)
	#final_data = new_model.predict(image,verbose=0)
	#print(final_data)
	final_data = final_data.item()
	final_data2 = final_data2.item()
	
	#if final_data == 0:
	print(f"original_file : {filepaths}, resultado_VGG {final_data} resultado Resnet {final_data2}")
	n+=1

print(f"total: {n}")

	
