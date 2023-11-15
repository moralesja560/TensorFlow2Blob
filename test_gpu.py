# RUNS WITH TENSOR ENVIRONMENT

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys
from urllib.request import Request, urlopen
import json
from urllib.parse import quote
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
token_Tel = os.getenv('TOK_EN_BOT')
Jorge_Morales = os.getenv('JORGE_MORALES')
Paintgroup = os.getenv('PAINTLINE')

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

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


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

s = tf.test.is_built_with_cuda()
now = datetime.now()
dt_string = now.strftime("%d%m%y-%H%M%S")
#send_message(Jorge_Morales,quote(f"Reporte de CUDA: \nSu computadora entregó este resultado \n{s}: {dt_string}"),token_Tel)


gchs_hora = 72
hour = 10
gchs_dia = 831
todai = 15
g_llena_h =22
g_llena_d = 628
g_vacia_h =50
g_vacia_d = 203
g_caidas = 79 - gchs_hora
pzas_g_caidas = (79-gchs_hora)*20

send_message(Paintgroup,quote(f"Producción de hora {hour}-{hour+1}: \nMeta Hora: 79 / 1,580 pz \n(-) Gch perdidas {(g_caidas)} / {(pzas_g_caidas):,} pz. \n(-)Huecos: {g_vacia_h} / {(g_vacia_h*20):,} pz. \n(=)Produccion: {g_llena_h} / {(g_llena_h*20):,} pz."),token_Tel)
send_message(Paintgroup,quote(f"Acumulado Hoy: \nMeta {((hour+1)*79*20):,} pz. \n(-) Pz x Gch perd {((hour+1)*79*20)-(gchs_dia*20):,} pz. \n(-) Huecos: {(g_vacia_d*20):,} pz. \n(=) Produccion: {(g_llena_d*20):,} pz. \nEficiencia Linea {((g_llena_d*20)/(gchs_dia*20)):.2%} \nEstimación para el final del día: {(((g_llena_d*20)/(hour+1))*24):,.0f} pz."),token_Tel)
