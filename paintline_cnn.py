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

RTSP_URL = 'rtsp://root:mubea@10.65.68.2:8554/axis-media/media.amp'
DELAY_SENSOR = 25.4
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
FRAME_SHAPE = 256
#Pandas DataFrame dictionaries
pd_dict = {'timestamp' : ['0'], 'G hora' : ['0'],'Hora' : ['0'],'G dia' : ['0'],'Dia' : ['0'],'G llena h' : ['0'],'G llena d' : ['0'],'G vacia h' : ['0'], 'G vacia d' : ['0']}
pd_dict2 = {'timestamp' : ['0']}


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
		#send_message(Paintgroup,quote(f"Reporte de Hora {hour}-{hour+1}: \nTotal gancheras: {gchs_hora}. \nHuecos: {g_vacia_h} \nLlenas: {g_llena_h} "),token_Tel)
		#send_message(Paintgroup,quote(f"Producción de hora {hour}-{hour+1}: \nTotal Hora: {(gchs_hora*20):,} pz. \nPzs Huecos: {(g_vacia_h*20):,} pz. \nProduccion: {(g_llena_h*20):,} pz."),token_Tel)
		if gchs_hora < 79:
			g_caidas = 79 - gchs_hora
			pzas_g_caidas = (79 - gchs_hora) *20
		else:
			pzas_g_caidas = 0
			g_caidas = 0
		#send_message(Paintgroup,quote(f"Producción de hora {hour}-{hour+1}: \nTotal Hora: {gchs_hora} / {(gchs_hora*20):,} pz. \nGch perdidas {(g_caidas)} / {(pzas_g_caidas):,} pz. \nHuecos: {g_vacia_h} / {(g_vacia_h*20):,} pz. \nProduccion: {g_llena_h} / {(g_llena_h*20):,} pz."),token_Tel)
		send_message(Paintgroup,quote(f"Producción de hora {hour}-{hour+1}: \nMeta Hora: 79 / 1,580 pz \n(-) Gch perdidas {(g_caidas)} / {(pzas_g_caidas):,} pz. \n(-) Huecos: {g_vacia_h} / {(g_vacia_h*20):,} pz. \n(=)Produccion: {g_llena_h} / {(g_llena_h*20):,} pz."),token_Tel)
		send_message(Paintgroup,quote(f"Acumulado Hoy: \nMeta {((hour+1)*79*20):,} pz. \n(-) Pz x Gch perd {((hour+1)*79*20)-(gchs_dia*20):,} pz. \n(-) Huecos: {(g_vacia_d*20):,} pz. \n(=) Produccion: {(g_llena_d*20):,} pz. \nEficiencia Linea {((g_llena_d*20)/((hour+1)*79*20)):.2%} \nEst. para final del día: {(((g_llena_d*20)/(hour+1))*24):,.0f} pz."),token_Tel)
		#se escribe el reporte
		print(gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d)
		write_log(gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d)
		#reset a la variable
		gchs_hora = 0
		g_vacia_h = 0
		g_llena_h = 0
		#se actualiza la hora
		hour = int(now.strftime("%H"))
		
					
	if todai != int(now.strftime("%d")):
		#send_message(Paintgroup,quote(f"Reporte del día {todai}: {gchs_dia}"),token_Tel)
		#send_message(Paintgroup,quote(f"Reporte del día {todai} : Cap Instalada: 1896 gch \nTotal gancheras: {(gchs_dia):,}. \nGch perdidas {(1896-gchs_dia):,} \nHuecos: {g_vacia_d} \nLlenas: {g_llena_d} "),token_Tel)
		send_message(Paintgroup,quote(f"Reporte de Producción día: {todai}: \nPzs perdidas {((1896-gchs_dia)*20):,} \nPzs Huecos: {(g_vacia_d*20):,} \nProduccion: {(g_llena_d*20):,} "),token_Tel)
		#reset a la variable
		gchs_dia = 0
		g_vacia_d = 0
		g_llena_d = 0
		#se actualiza la hora
		todai = int(now.strftime("%d"))
	return gchs_hora,hour,gchs_dia,todai,g_llena_h,g_llena_d,g_vacia_h,g_vacia_d

def retrieve_img():
	cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
	y=0
	x=0
	h=720 #1080
	w=920 #1350
	for x in range(3):
		success, frame = cap.read()
		try:
			crop = frame[y:y+h, x:x+w]
		except:
			#cap.release()
			#return None,1
			print(f"intento {x}")
			continue
		else:
			cap.release()
			return crop,0
	return None,1
	cap.release()

def write_log_gch():
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
	#print (dt_string)
	#print("date and time =", dt_string)	
	mis_docs = My_Documents(5)
	#ruta = str(mis_docs)+ r"\paintline.txt"
	pd_ruta = str(mis_docs)+ r"\paintline_missing_df.csv"
	#file_exists = os.path.exists(ruta)
	pd_file_exists = os.path.exists(pd_ruta)
	#check if pandas DataFrame exists to load the stuff or to create with dummy data.
	if pd_file_exists:
		pd_log = pd.read_csv(pd_ruta)
	else:
		pd_log = pd.DataFrame(pd_dict2)

	new_row = {'timestamp' : [dt_string]}
	new_row_pd = pd.DataFrame(new_row)
	pd_concat = pd.concat([pd_log,new_row_pd])
	pd_concat.to_csv(pd_ruta,index=False)

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
		before_gch = 0
		#carga el modelo
		start11 = time.time()
		print("loading model")
		new_model = tf.keras.models.load_model(resource_path(r"resnet_paint"))
		end11 = time.time()
		print(f"NN Load time {(end11 - start11):.3f}")
		#inicialización de PLC
		while True:
			try:
				#sensor
				plc.open()
				#handles
				hanger_counter = plc.get_handle('.I_Hang_Counter')
				var_handle_actual_hook = plc.get_handle('SCADA.This_hook')
				var_handle_full_hr = plc.get_handle('SCADA.Full_hooks_hr')
				var_handle_full_day = plc.get_handle('SCADA.Full_hooks_day')
				var_handle_empty_hr = plc.get_handle('SCADA.Empty_hooks_hr')
				var_handle_empty_day = plc.get_handle('SCADA.Empty_hooks_day')
				#estado del sensor
				cell1 = plc.read_by_name("", plc_datatype=pyads.PLCTYPE_BOOL,handle=hanger_counter)
				if cell1:
					start10 = time.time()
					#llenas por h
					plc.write_by_name("", gch_llena_h, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_full_hr)
					#llenas por d
					plc.write_by_name("", gch_llena_d, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_full_day)
					#vacias por h
					plc.write_by_name("", gch_vacia_h, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_empty_hr)
					#vacias por d
					plc.write_by_name("", gch_vacia_d, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_empty_day)	
					end10 = time.time()
					print(f"PLC write time {(end10 - start10):.3f}")

			except Exception as e:
				#send_message(Jorge_Morales,quote(f"Falla de app: {e}. Si es el 1861, por favor conectarse al PLC via Twincat System Manager. Con eso se hace la conexión ADS"),token_Tel)
				print(e)
				try:
					plc.release_handle(hanger_counter)
					plc.release_handle(var_handle_actual_hook)
					plc.release_handle(var_handle_full_hr)
					plc.release_handle(var_handle_full_day)
					plc.release_handle(var_handle_empty_hr)
					plc.release_handle(var_handle_empty_day)
					plc.close()
				except:
					print("couldnt close")
				finally:
					break
			else:
				#Funcion de reporte de hora y dia
				contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d = send_reports(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
				if cell1:
					if before_gch == 0:
						before_gch = time.time()
						start = 0
					else:
						start = time.time()
						print(int(start - before_gch))
						if (start - before_gch) > 47:
							print("ganchera timeout")
							write_log_gch()
						before_gch = start


					print(f"sensor received")
					now = datetime.now()
					times = now.strftime("%d%m%y-%H%M%S")
					time.sleep(DELAY_SENSOR)
					crop,result = retrieve_img()
					if result == 1:
						print("No hay imagen disponible")
						send_message(Jorge_Morales,quote(f"No se pudo la imagen en Pintura."),token_Tel)
						state_save(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
						gch_llena_h +=1
						gch_llena_d +=1
						plc.write_by_name("", 1, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_actual_hook)	
						continue
					else:
						end1 = time.time()
						print(f"image processed {(end1 - start - DELAY_SENSOR):.3f}")
						#Aqui es donde movemos el asunto
						# PASO 1: IMAGEN SE REDUCE A 256X256
						image = cv2.resize(crop,dsize=(FRAME_SHAPE,FRAME_SHAPE), interpolation = cv2.INTER_CUBIC) 
						# METEMOS A LA RED NEURONAL
						final_data = new_model.predict(np.expand_dims(image, axis=0),verbose=0)
						end2 = time.time()
						print(f"NN finished {(end2 - end1):.3f}")
						#SACAMOS LA INFO DE LA PREDICCION
						final_data = final_data.item()
						#GUARDAMOS LA IMAGEN
						# SI ES MAYOR A -0.5, ENTONCES LA gch ESTA LLENA
						contador_gchs +=1
						contador_gchs_day +=1
						if final_data >= -0.9:
							cv2.imwrite(resource_path(f'resources/full/F{final_data}.{times}.jpg'), crop)
							print(f"full image stored with {int(final_data)}")
							gch_llena_h +=1
							gch_llena_d +=1
							plc.write_by_name("", 1, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_actual_hook)	
						else:
							cv2.imwrite(resource_path(f'resources/empty/V{final_data}.{times}.jpg'), crop)
							print(f"empty image stored with {int(final_data)}")
							gch_vacia_h +=1
							gch_vacia_d +=1
							plc.write_by_name("", 2, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_actual_hook)	
						#ACTUALIZAMOS LOS CONTADORES.
						#contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d = send_reports(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
						state_save(contador_gchs,hora,contador_gchs_day,day,gch_llena_h,gch_llena_d,gch_vacia_h,gch_vacia_d)
						end3 = time.time()
						print(f"Waiting: Last Cycle Time {(end3 - start):.3f}")



			if self._stop_event.is_set():
				# close connection
				print("saliendo")
				plc.release_handle(hanger_counter)
				plc.release_handle(var_handle_actual_hook)
				plc.release_handle(var_handle_full_hr)
				plc.release_handle(var_handle_full_day)
				plc.release_handle(var_handle_empty_hr)
				plc.release_handle(var_handle_empty_day)
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