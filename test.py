import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import AUC
from keras.models import load_model
import os
import numpy as np
import cv2
from PIL import Image

loaded_model = tf.keras.models.load_model('ultrasound_birad_model.hdf5', compile=False)

loaded_model.compile(optimizer=Adam(learning_rate=0.0007), loss='categorical_crossentropy', metrics=['accuracy'])

image = Image.open(r'test_inst\birads - 2 (3).bmp')
image = image.convert('L')
image_raw = np.array(image)

crop2 = cv2.resize(image_raw, (1000, 1000))
crop2 = np.stack((crop2,) * 3, axis=-1)
crop2 = np.expand_dims(crop2, axis=0)

prediction = loaded_model.predict(crop2)
print(prediction)