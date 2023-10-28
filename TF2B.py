import blobconverter
import os,sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


blob_path = blobconverter.from_tf(
    frozen_pb=resource_path(r"resnet_pizza\saved_model.pb"),
    data_type="FP16",
    shaves=6,
    optimizer_params=[
        "--reverse_input_channels",
        "--input_shape=[1,256,256,3]",
        "--input=1:mul_1",
        "--output=ArgMax",
    ],
)