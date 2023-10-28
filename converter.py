from openvino.tools import mo
import os,sys
from openvino.runtime import serialize



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#Variables
path = resource_path(r"resnet_pizza")
input_shape = [1,256,256,3]
layout = "nhwc"
values = [127.5,127.5,127.5]
scale = [127.5]
output_path = r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\TensorFlow2Blob\pizza.xml'

finished_model = mo.convert_model(saved_model_dir=path,input_shape=input_shape,source_layout=layout, mean_values=values, scale_values=scale, reverse_input_channels=True,progress=True)
serialize(finished_model, xml_path=output_path)

