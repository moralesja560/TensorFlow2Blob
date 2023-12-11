# RUNS WITH TENSOR ENVIRONMENT

from random import shuffle
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense,Flatten, Conv2D,MaxPool2D,Activation
from keras import Sequential
from tensorflow import keras
import os, sys
import matplotlib.pyplot as plt
from datetime import datetime
from keras.applications import ResNet101

tf.random.set_seed(42)
#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#This funcion plots curves

def plot_loss_curves(history):
  val_acc =history.history["val_accuracy"]
  val_loss= history.history["val_loss"]

  train_acc = history.history["accuracy"]
  train_loss = history.history["loss"]
  epochs = range(len(history.history["loss"]))

  plt.plot(epochs,train_loss,label="training loss")
  plt.plot(epochs,val_loss,label="val loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()
  
  plt.plot(epochs,train_acc,label="training accuracy")
  plt.plot(epochs,val_acc,label="val accuracy")
  plt.title("Accuracy")
  plt.xlabel("epochs")
  plt.legend()
  plt.ylim([0,1])
  plt.show()

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

tf.random.set_seed(42)

resnet = ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3),
    pooling=None
)
resnet.trainable = False

#set the directory path WORK COMPUTER
train_dir = r"E:\tensor\train"
test_dir = r"E:\tensor\test"

print(train_dir)
##-----------------------The baseline model--------------------------#

def baseline_model(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	train_data = train_datagen.flow_from_directory(
    	directory=train_dir,
    	batch_size=batch_size,
    	target_size=(224,224),
    	class_mode = 'binary',
    	seed=42
	)

	valid_data = valid_datagen.flow_from_directory(
	    directory=test_dir,
	    batch_size=batch_size,
	    target_size=(224,224),
	    class_mode = 'binary',
	    seed=42
	)
	learning_rate_calc = (0.1 *(batch_size)/256)

	model_base = tf.keras.models.Sequential([
	Conv2D(filters=10, kernel_size=3, strides=(1,1),padding= 'valid', activation = "relu", input_shape =(224,224,3)),
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"dogcatmodel"), monitor='accuracy',save_best_only= True,save_weights_only=False,verbose=1)
	early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=2,verbose=1,mode='max')
	#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 * ( 1 + math.cos( epoch * (math.pi))/(594)))

	data_model = model_base.fit(train_data,epochs=epochs,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[early_cb],verbose=1)
	return data_model

def model_2(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	train_data = train_datagen.flow_from_directory(
    	directory=train_dir,
    	batch_size=batch_size,
    	target_size=(224,224),
    	class_mode = 'binary',
    	seed=42
	)

	valid_data = valid_datagen.flow_from_directory(
	    directory=test_dir,
	    batch_size=batch_size,
	    target_size=(224,224),
	    class_mode = 'binary',
	    seed=42
	)
	learning_rate_calc = (0.1 *(batch_size)/256)

	model_base = tf.keras.models.Sequential([

	tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"dogcatmodel_2"), monitor='val_accuracy',save_best_only=True,save_weights_only=False,verbose=1)
	#early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=3,verbose=1,mode='max')
	#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 * ( 1 + math.cos( epoch * (math.pi))/(594)))

	data_model = model_base.fit(train_data,epochs=epochs,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[cp_callback],verbose=1,use_multiprocessing=True, workers=8)

def model_3(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	train_data = train_datagen.flow_from_directory(
    	directory=train_dir,
    	batch_size=batch_size,
    	target_size=(256,256),
    	class_mode = 'binary',
		shuffle=True,
    	seed=42
	)

	valid_data = valid_datagen.flow_from_directory(
	    directory=test_dir,
	    batch_size=batch_size,
	    target_size=(256,256),
	    class_mode = 'binary',
	    seed=42
	)
	learning_rate_calc = (0.1 *(batch_size)/256)

	model_base = tf.keras.models.Sequential([

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256,256, 3)),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	#tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	#tf.keras.layers.Dropout(0.2),
	
	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	#tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),
	
	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),

	tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),
	
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),

	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),

	tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
	tf.keras.layers.MaxPooling2D((2,2), padding='same'),

	tf.keras.layers.Flatten(),
	#tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
	#tf.keras.layers.Dropout(0.5),
	#tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"dogcatmodel_alt"), monitor='val_accuracy',save_best_only=True,save_weights_only=False,verbose=1)
	#early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=3,verbose=1,mode='max')
	#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 * ( 1 + math.cos( epoch * (math.pi))/(594)))

	data_model = model_base.fit(train_data,epochs=epochs,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[cp_callback],verbose=1, workers=2)
	return data_model


def model_4(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	train_data = train_datagen.flow_from_directory(
    	directory=train_dir,
    	batch_size=batch_size,
    	target_size=(256,256),
    	class_mode = 'binary',
		shuffle=True,
    	seed=42
	)

	valid_data = valid_datagen.flow_from_directory(
	    directory=test_dir,
	    batch_size=batch_size,
	    target_size=(256,256),
	    class_mode = 'binary',
	    seed=42
	)

	model = tf.keras.models.Sequential([ 
		tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(256,256,3)),
		tf.keras.layers.MaxPooling2D(2,2),
    	tf.keras.layers.BatchNormalization(),
      	tf.keras.layers.Dropout(0.2),

      	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
      	tf.keras.layers.MaxPooling2D(2,2),
      	tf.keras.layers.Dropout(0.2),

      	tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
      	tf.keras.layers.MaxPooling2D(2,2),
      	tf.keras.layers.Dropout(0.2),

      	tf.keras.layers.Flatten(),
      	tf.keras.layers.Dense(256,activation='relu'),
      	tf.keras.layers.Dropout(0.5),
      	tf.keras.layers.Dense(128,activation='relu'),
		tf.keras.layers.Dense(1,activation='sigmoid')  
	])

	model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(),metrics=[keras.metrics.BinaryAccuracy()])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"cnn_normal_paintline"), monitor='binary_accuracy',save_best_only=True,save_weights_only=False,verbose=1)
	#early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=3,verbose=1,mode='max')
	#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 * ( 1 + math.cos( epoch * (math.pi))/(594)))

	data_model = model.fit(train_data,epochs=epochs,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[cp_callback],verbose=1, workers=8)
	return data_model

def model_5(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	ds = keras.utils.image_dataset_from_directory(train_dir,
            batch_size=batch_size,
            image_size=(256,256),
            shuffle=True)
	
	data_augmentation = keras.Sequential(
    [
       # keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.3),
		keras.layers.RandomFlip(mode="horizontal",seed=42),
        keras.layers.Rescaling(scale=1./127.5, offset=-1),
    ]
	)

	ds_test = keras.utils.image_dataset_from_directory(test_dir,
            batch_size=16,
            image_size=(256,256),
            shuffle=True)
	
	# Model Input
	input = keras.layers.Input(shape=(256,256, 3))
	# Data augmentation
	x = data_augmentation(input)
	# resnet
	x = resnet(input)
	# Global pooling
	x = keras.layers.GlobalAveragePooling2D()(x)
	# dropout正则化
	x = keras.layers.Dropout(0.2)(x)
	# 输出层
	outputs = keras.layers.Dense(1)(x)

	# 模型构建
	model = keras.models.Model(input, outputs)

	model.summary()
	model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()])
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"resnet_paint_2"), monitor='binary_accuracy',save_best_only=True,save_weights_only=False,verbose=1)
	model.fit(ds, epochs=epochs,validation_data=ds_test,validation_steps=len(ds_test),callbacks=[cp_callback],verbose=1, workers=8)

def model_6(train_dir,test_dir,train_datagen,valid_datagen,epochs,batch_size):
	train_data = train_datagen.flow_from_directory(
    	directory=train_dir,
    	batch_size=batch_size,
    	target_size=(224,224),
    	class_mode = 'binary',
    	seed=42
	)

	valid_data = valid_datagen.flow_from_directory(
	    directory=test_dir,
	    batch_size=batch_size,
	    target_size=(224,224),
	    class_mode = 'binary',
	    seed=42
	)


	model_base = tf.keras.models.Sequential([
	Conv2D(filters=10, kernel_size=3, strides=(1,1),padding= 'valid', activation = "relu", input_shape =(224,224,3)),
	Conv2D(filters=10, kernel_size=3, activation = "relu"),
	tf.keras.layers.MaxPool2D(pool_size=2,padding="valid"),
	tf.keras.layers.Conv2D(10,3,activation="relu"),
	tf.keras.layers.Conv2D(10,3,activation="relu"),
	tf.keras.layers.MaxPool2D(pool_size=2,padding="valid"),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"model_7_paintline"), monitor='accuracy',save_best_only= True,save_weights_only=False,verbose=1)
	#early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=2,verbose=1,mode='max')
	#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 * ( 1 + math.cos( epoch * (math.pi))/(594)))

	data_model = model_base.fit(train_data,epochs=epochs,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[cp_callback],verbose=1)
	return data_model

def data_transformation(zoom,shear,flip_v,flip_h,rotation,w_shift,h_shift):
	train_datagen = ImageDataGenerator(
    	rescale=1./255,
    	rotation_range=rotation,
    	shear_range=shear,
    	zoom_range=zoom,
    	width_shift_range=w_shift,
    	height_shift_range=h_shift,
    	horizontal_flip=flip_h,
    	vertical_flip=flip_v,
	)
    
	valid_datagen = ImageDataGenerator(rescale=1./255)
	return train_datagen,valid_datagen


if __name__ == '__main__':
	print(datetime.now())
	s = tf.test.is_built_with_cuda()
	print(f"Reporte de CUDA: {s}")
	train_datagen_f,valid_datagen_f = data_transformation(zoom=0.1,shear=0.1,flip_h=False,flip_v=False,rotation=0.1,w_shift=0.1,h_shift=0.1)
	#model_tr_data = baseline_model(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f,epochs=30,batch_size=16)
	#plot_loss_curves(model_tr_data)
	#model_tr_data2 = model_2(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f,epochs=50,batch_size=100)
	#plot_loss_curves(model_tr_data2)
	#model_tr_data3 = model_3(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f,epochs=50,batch_size=16)
	#plot_loss_curves(model_tr_data3)
	#rain_datagen_g,valid_datagen_g = data_transformation(zoom=0.2,shear=0.2,flip_h=True,flip_v=False,rotation=0.4,w_shift=0.2,h_shift=0.2)
	#model_tr_data4 = model_4(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_g,valid_datagen=valid_datagen_g,epochs=50,batch_size=16)
	#model_tr_data5 = model_5(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f,epochs=5,batch_size=32)
	#plot_loss_curves(model_tr_data5)
	model_tr_data6 = model_6(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f,epochs=5,batch_size=32)
	plot_loss_curves(model_tr_data6)



"""
Flow chart

Model is overfitting? (training_accuracy>99 & val_accuracy<90)
	* if yes: use data augmentation or another technique to make the data harder to figure it out.
	* if not: improve model by altering hyperparameters or select more epochs.

baseline:
Epoch 16/30
1245/1245 [==============================] - 178s 143ms/step - loss: 0.4271 - accuracy: 0.8033 - val_loss: 0.4159 - val_accuracy: 0.8217

model_2:
Epoch 40/40
312/312 [==============================] - ETA: 0s - loss: 0.2802 - accuracy: 0.8880

A method recommended by Geoff Hinton is to add layers until you start to overfit your training set. Then you add dropout or another regularization method.

"""
	