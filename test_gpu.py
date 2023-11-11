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
send_message(Jorge_Morales,quote(f"Reporte de CUDA: \nSu computadora entregó este resultado \n{s}: {dt_string}"),token_Tel)