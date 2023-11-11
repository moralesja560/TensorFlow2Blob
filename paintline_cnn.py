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
import csv


load_dotenv()
token_Tel = os.getenv('TOK_EN_BOT')
Jorge_Morales = os.getenv('JORGE_MORALES')
Paintgroup = os.getenv('PAINTLINE')

RTSP_URL = 'rtsp://root:mubea@10.65.68.2/axis-media/media.amp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
FRAME_SHAPE = 256
#Pandas DataFrame dictionaries
pd_dict = {'timestamp' : ['0'], 'G hora' : ['0'],'Hora' : ['0'],'G dia' : ['0'],'Dia' : ['0'],'G llena h' : ['0'],'G llena d' : ['0'],'G vacia h' : ['0'], 'G vacia d' : ['0']}



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
			pass
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

def state_recover():
	ruta_state = resource_path("images/last_state.csv")
	file_exists = os.path.exists(ruta_state)
	if file_exists == True:
		pass
	else:
		with open(ruta_state,"w+") as f:
			f.write(f"{int(0)},{int(0)},{int(0)},{int(0)},{int(0)},{int(0)},{int(0)},{int(0)}")		

	with open(resource_path("images/last_state.csv")) as file:
		type(file)
		csvreader = csv.reader(file)
		rows2 = []
		for row in csvreader:
			rows2.append(row)
		
		contador_h = int(rows2[0][0])
		hora = int(rows2[0][1])
		contador_d = int(rows2[0][2])
		day = int(rows2[0][3])
		gch_llena_h  = int(rows2[0][4])
		gch_llena_d = int(rows2[0][5])
		gch_vacia_h = int(rows2[0][6])
		gch_vacia_d = int(rows2[0][7])

		return contador_h,hora,contador_d,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d

def state_save(fmb46_state,hora,contador_d,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d):
	ruta_state = resource_path("images/last_state.csv")

	with open(ruta_state, "w+") as file_object:

		file_object.write(f"{fmb46_state},{hora},{contador_d},{day},{gch_llena_h},{gch_llena_d},{gch_vacia_h},{gch_vacia_d}")

def send_reports(gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d):
	#hourly report
	now = datetime.now()
	if hour != int(now.strftime("%H")):
		#send_message(Paintgroup,quote(f"Reporte de Hora: {hour}-{hour+1}: {gchs_hora}"),token_Tel)
		send_message(Paintgroup,quote(f"Reporte de Hora: {hour}-{hour+1}: \nTotal gancheras: {gchs_hora}. \nHuecos: {g_vacia_h} \nLlenas: {g_llena_h} "),token_Tel)
		send_message(Paintgroup,quote(f"Estimado de Producción: {hour}-{hour+1}: \nTotal Hora: {gchs_hora*20}. \nPzs Perdidas: {g_vacia_h*20} \nProduccion: {g_llena_h*20} "),token_Tel)
		#reset a la variable
		gchs_hora = 0
		g_vacia_h = 0
		g_llena_h = 0
		#se actualiza la hora
		hour = int(now.strftime("%H"))
		#se escribe el reporte
		write_log(gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d)
					
	if todai != int(now.strftime("%d")):
		#send_message(Paintgroup,quote(f"Reporte del día {todai}: {gchs_dia}"),token_Tel)
		send_message(Paintgroup,quote(f"Reporte del día {todai} : \nTotal gancheras: {gchs_dia}. \nHuecos: {g_vacia_d} \nLlenas: {g_llena_d} "),token_Tel)
		send_message(Paintgroup,quote(f"Estimado de Producción día: {todai}: \nTotal: {gchs_dia*20}. \nPzs Perdidas: {g_vacia_d*20} \nProduccion: {g_llena_d*20} "),token_Tel)
		#reset a la variable
		gchs_dia = 0
		g_vacia_d = 0
		g_llena_d = 0
		#se actualiza la hora
		todai = int(now.strftime("%d"))
	return gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d

def retrieve_img():
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
		cap.release()
		return None
	else:
		cap.release()
		return crop


def write_log(gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d):
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
	#print (dt_string)
	#print("date and time =", dt_string)	
	mis_docs = My_Documents(5)
	#ruta = str(mis_docs)+ r"\paintline.txt"
	pd_ruta = str(mis_docs)+ r"\paintlinehr_df.csv"
	#file_exists = os.path.exists(ruta)
	pd_file_exists = os.path.exists(pd_ruta)
	"""
	if file_exists == True:
		with open(ruta, "a+") as file_object:
			# Move read cursor to the start of file.
			file_object.seek(0)
			# If file is not empty then append '\n'
			data = file_object.read(100)
			if len(data) > 0 :
				file_object.write("\n")
				file_object.write(f" timestamp {dt_string}: Cell1: {cell1}")
	else:
		with open(ruta,"w+") as f:
				f.write(f" timestamp {dt_string}: Cell1: {cell1}")
	"""
	#check if pandas DataFrame exists to load the stuff or to create with dummy data.
	if pd_file_exists:
		pd_log = pd.read_csv(pd_ruta)
	else:
		pd_log = pd.DataFrame(pd_dict)

	new_row = {'timestamp' : [dt_string], 'G hora' : [gchs_hora],'Hora' : [hour],'G dia' : [gchs_dia],'Dia' : [todai],'G llena h' : [g_llena_h],'G llena d' : [g_llena_d],'G vacia h' : [g_vacia_h], 'G vacia d' : [g_vacia_d]}
	new_row_pd = pd.DataFrame(new_row)
	pd_concat = pd.concat([pd_log,new_row_pd])
	pd_concat.to_csv(pd_ruta,index=False)
##--------------------the thread itself--------------#

class hilo1(threading.Thread):
	#thread init procedure
	# i think we pass optional variables to use them inside the thread
	def __init__(self):
		threading.Thread.__init__(self)
		self._stop_event = threading.Event()
	#the actual thread function
	def run(self):
		#arranca con los datos guardados
		contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d = state_recover()
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
				#if state_hanger != cell1:		
				#	print(cell1)
				#	state_hanger = cell1
				#Funcion de reporte de hora y dia
				contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d = send_reports(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
				write_log(cell1)
				if cell1:
					print("sensor received")
					contador_gchs +=1
					contador_gchs_day +=1
					#st = time.time()
					now = datetime.now()
					times = now.strftime("%d%m%y-%H%M%S")
					time.sleep(25.4)
					crop = retrieve_img()
					if crop == None:
						print("No hay imagen disponible")
						state_save(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
						continue
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
						#GUARDAMOS LA IMAGEN
						# SI ES MAYOR A 1, ENTONCES LA gch ESTA LLENA
						if final_data >= 1:
							cv2.imwrite(resource_path(f'resources/full/F{final_data}.{times}.jpg'), crop)
							print(f"full image stored with {int(final_data)}")
							gch_llena_h +=1
							gch_llena_d +=1
						else:
							cv2.imwrite(resource_path(f'resources/empty/V{final_data}.{times}.jpg'), crop)
							print(f"empty image stored with {int(final_data)}")
							gch_vacia_h +=1
							gch_vacia_d +=1
						#ACTUALIZAMOS LOS CONTADORES.
						state_save(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)



			if self._stop_event.is_set():
				# close connection
				print("saliendo")
				plc.release_handle(var_handle46_1)
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