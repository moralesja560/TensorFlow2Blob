# RUNS WITH OPENVINO ENVIRONMENT
import blobconverter
import os,sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


blobconverter.from_openvino(xml=resource_path(r"pizza.xml"),bin=resource_path(r"pizza.bin"),data_type="FP16",shaves=6)