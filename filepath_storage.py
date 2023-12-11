import cv2
from datetime import datetime
import os


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



def NN_storage(img,command):
	"""
	This function is to store images in the correct folder to ensure trazability
	The folder is in Mydocuments
	Documents/<day>/<hour>/full
	Documents/<day>/<hour>/empty
	"""
	# a new image has been received, check if folder exists
	# Step 1: path first step
	mis_docs = My_Documents(5)
	now = datetime.now()
	#date
	NN_folder_name= "\\Paintline_Evidence\\" + now.strftime("%d-%m-%y--%H")
	pd_ruta = str(mis_docs) + NN_folder_name
	pd_ruta_full = pd_ruta + r'\\full'
	pd_ruta_empty = pd_ruta + r'\\empty'
	
	if not os.path.isdir(pd_ruta):
		os.makedirs(pd_ruta)
		os.makedirs(pd_ruta_full)
		os.makedirs(pd_ruta_empty)
	
	if command == 1:
		#full hanger
		times = now.strftime("%d%m%y-%H%M%S")
		cv2.imwrite(pd_ruta_full+f"\{times}.jpg",img)
	elif command == 2:
		times = now.strftime("%d%m%y-%H%M%S")
		cv2.imwrite(pd_ruta_empty+f"\{times}.jpg",img)		


if __name__ == '__main__':
	img = cv2.imread(r"E:\temp.jpg")
	NN_storage(img,2)