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

FRAME_SHAPE = 224

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


file_list = os.listdir(r"E:\tensor\train\full")
n=0
for filepaths in file_list:
	ruta_c = r"E:\tensor\train\full"+"\\"+filepaths
	print(f"{ruta_c}")
	img = cv2.imread(ruta_c)
	y=0
	x=0
	h=720 #1080
	w=920 #1350
	crop = img[y:y+h, x:x+w]
	cv2.imwrite(ruta_c,crop)

print(f"total: {n}")

	