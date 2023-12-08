# RUNS WITH TENSOR ENVIRONMENT

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf




if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

s = tf.test.is_built_with_cuda()
print(f"Reporte de CUDA: \nSu computadora entregó este resultado \n{s}")

