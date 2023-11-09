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

# define a video capture object
vid = cv2.VideoCapture(0)
FRAME_SHAPE = 256

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

new_model = tf.keras.models.load_model(resource_path(r"resnet_paint"))

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
while True:
	ret, img = vid.read()
	image = cv2.resize(img,dsize=(FRAME_SHAPE,FRAME_SHAPE), interpolation = cv2.INTER_CUBIC) 
	final_data = new_model.predict(np.expand_dims(image, axis=0),verbose=1)
	#final_data = new_model.predict(image,verbose=0)
	#print(final_data)
	final_data = final_data.item()
	
	if final_data <-3:
		print(f"It's a pizza {final_data}")
		cv2.putText(img, 'PIZZA', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
	elif final_data> 0.70 and final_data < 10.99:
		print(f"It's a Steak {final_data}")
		cv2.putText(img, 'STEAK', (10,450), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow("Data",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()