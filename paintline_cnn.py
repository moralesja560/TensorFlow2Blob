import pyads
import sys
import threading
import time
from dotenv import load_dotenv
import os
from urllib.request import Request, urlopen
import json
from urllib.parse import quote
import pandas as pd
from datetime import datetime
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from fileinput import filename
import tensorflow as tf


load_dotenv()
token_Tel = os.getenv('TOK_EN_BOT')
Jorge_Morales = os.getenv('JORGE_MORALES')
Paintgroup = os.getenv('PAINTLINE')
RTSP_URL = 'rtsp://root:mubea@10.65.68.2/axis-media/media.amp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
FRAME_SHAPE = 256



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def My_Documents(location):
	import ctypes.wintypes
		#####-----This section discovers My Documents default path --------
		#### loop the "location" variable to find many paths, including AppData and ProgramFiles
	CSIDL_PERSONAL = location       # My Documents
	SHGFP_TYPE_CURRENT = 0   # Get current, not default value
	buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
	ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
	#####-------- please use buf.value to store the data in a variable ------- #####
	#add the text filename at the end of the path
	temp_docs = buf.value
	return temp_docs

def send_message(user_id, text,token):
	global json_respuesta
	url = f"https://api.telegram.org/{token}/sendMessage?chat_id={user_id}&text={text}"
	#resp = requests.get(url)
	#hacemos la petición
	try:
		ruta_state = resource_path("images/tele.txt")
		file_exists = os.path.exists(ruta_state)
		if file_exists == False:
			return
		else:
			respuesta  = urlopen(Request(url))
	except Exception as e:
		print(f"Ha ocurrido un error al enviar el mensaje: {e}")
	else:
		#recibimos la información
		cuerpo_respuesta = respuesta.read()
		# Procesamos la respuesta json
		json_respuesta = json.loads(cuerpo_respuesta.decode("utf-8"))
		print("mensaje enviado exitosamente")

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

##--------------------the thread itself--------------#

class hilo1(threading.Thread):
	#thread init procedure
	# i think we pass optional variables to use them inside the thread
	def __init__(self):
		threading.Thread.__init__(self)
		self._stop_event = threading.Event()
	#the actual thread function
	def run(self):
		print("loading model")
		new_model = tf.keras.models.load_model(resource_path(r"resnet_paint"))
		print("model_loaded")
		#inicialización de PLC
		while True:
			try:
				#sensor
				plc.open()
				var_handle46_1 = plc.get_handle('.I_Hang_Counter')
				cell1 = plc.read_by_name("", plc_datatype=pyads.PLCTYPE_BOOL,handle=var_handle46_1)
			except Exception as e:
				#send_message(Jorge_Morales,quote(f"Falla de app: {e}. Si es el 1861, por favor conectarse al PLC via Twincat System Manager. Con eso se hace la conexión ADS"),token_Tel)
				print(e)
				try:
					plc.release_handle(var_handle46_1)
					plc.close()
				except:
					print("couldnt close")
				finally:
					break
			else:
				if cell1:
					print("sensor received")
					st = time.time()
					time.sleep(25.4)
					print("processing image")
					cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
					_, frame = cap.read()
					y=0
					x=0
					h=1080
					w=1350
					try:
						crop = frame[y:y+h, x:x+w]
					except:
						pass
					else:
						#et = time.time()
						#ts = str(et - st-1)
						#Aqui es donde movemos el asunto
						# PASO 1: IMAGEN SE REDUCE A 256X256
						image = cv2.resize(crop,dsize=(FRAME_SHAPE,FRAME_SHAPE), interpolation = cv2.INTER_CUBIC) 
						# METEMOS A LA RED NEURONAL
						final_data = new_model.predict(np.expand_dims(image, axis=0),verbose=1)
						#SACAMOS LA INFO DE LA PREDICCION
						final_data = final_data.item()
						#print(resource_path(f'resources/img{final_data}.jpg'))
						
						#GUARDAMOS LA IMAGEN
						# SI ES MAYOR A 1, ENTONCES LA GANCHERA ESTA LLENA
						if final_data >= 1:
							cv2.imwrite(resource_path(f'resources/full/llena{final_data}.jpg'), crop)
							print(f"full image stored with {int(final_data)}")
						else:
							cv2.imwrite(resource_path(f'resources/empty/vacia{final_data}.jpg'), crop)
							print(f"empty image stored with {int(final_data)}")
						#ACTUALIZAMOS LOS CONTADORES.



			if self._stop_event.is_set():
				# close connection
				print("saliendo")
				plc.release_handle(var_handle46_1)
				cap.release()
				plc.close()
				break
	def stop(self):
		self._stop_event.set()

	def stopped(self):
		return self._stop_event.is_set()
#----------------------end of thread 1------------------#

#---------------Thread 2 Area----------------------#
class hilo2(threading.Thread):
	#thread init procedure
	# i think we pass optional variables to use them inside the thread
	def __init__(self,thread_name,opt_arg):
		threading.Thread.__init__(self)
		self.thread_name = thread_name
		self.opt_arg = opt_arg
		self._stop_event = threading.Event()
	#the actual thread function
	def run(self):
		#check for thread1 to keep running
		while True:
			if [t for t in threading.enumerate() if isinstance(t, hilo1)]:
				try:
					time.sleep(5)
				except:
					self._stop_event.set()
			else:
				print(f"A problem occurred... Restarting Thread 1")
				time.sleep(4)
				thread1 = hilo1()
				thread1.start()
				print(f"Thread 1 Started")
			
			if self._stop_event.is_set() == True:
				print("Thread 2 Stopped")
				break

	def stop(self):
		self._stop_event.set()




if __name__ == '__main__':

	# connect to the PLC
	try:
		pyads.open_port()
		ams_net_id = pyads.get_local_address().netid
		print(ams_net_id)
		pyads.close_port()
		plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
		plc.set_timeout(1000)
	except:
		print("No se pudo abrir la conexión")
		sys.exit()
	 #open the connection
	else:
		pass
	thread1 = hilo1()
	thread1.start()
	thread2 = hilo2(thread_name="hilo2",opt_arg="h")
	thread2.start()
	while True:
		stop_signal = input()
		if stop_signal == "T":
			thread1.stop()
			thread2.stop()
		break