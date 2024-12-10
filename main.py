import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import DracoPy
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import layers, models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import keras
from torch import nn
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Multiply, add
from keras.initializers import HeNormal
from keras.losses import Huber
import tensorflow_addons as tfa

IMG_SIZE = 64

def compress(pc, compression_level=10, quantization=30):
    return DracoPy.encode(pc, compression_level=compression_level, quantization_bits=quantization)

def decompress(compressed):
    decompressed = DracoPy.decode(compressed).points
    return decompressed

def pc2img(pc):
    if len(pc.shape) == 2:
        pc = pc.reshape(1, pc.shape[0], pc.shape[1])

    shape = pc.shape

    z = pc[:, :, 2]
    indices = np.argsort(z, axis=1)
    pc = np.take_along_axis(pc, indices[..., None], axis=1)

    pc = pc.reshape(shape[0], IMG_SIZE, IMG_SIZE, 3)

    return pc

pcs = np.random.rand(10000, IMG_SIZE ** 2, 3).astype(np.float64)
imgs = pc2img(pcs)
imgs_compressed = [compress(pc, 5, 30) for pc in pcs]
imgs_decompressed = [decompress(img) for img in imgs_compressed]
imgs_decompressed = pc2img(np.array(imgs_decompressed, dtype=np.float64))
# normalize
imgs_decompressed -= np.min(imgs_decompressed)
imgs_decompressed /= np.max(imgs_decompressed)
imgs -= np.min(imgs)
imgs /= np.max(imgs)

print(f"imgs_decompressed shape: {imgs_decompressed.shape}")
print(f"imgs shape: {imgs.shape}")
print(f"Data type of imgs_decompressed: {imgs_decompressed.dtype}")
print(f"Data type of imgs: {imgs.dtype}")
print(f"MSE: {mean_squared_error(imgs.flatten(), imgs_decompressed.flatten())}")
print()

def build_model(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (1, 1), activation='linear', padding='same')  # Output layer
    ])
    return model

def scaled_mse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    variance = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
    scaled_mse_value = mse / (variance + 1e-32) # Add epsilon to avoid division by zero
    return scaled_mse_value

def train(imgs, imgs_decompressed):
    model = build_model()
    print('Created new model with shape:', model.input_shape)

    saver = ModelCheckpoint('model', monitor='val_mse', verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch')

    optimizer = keras.optimizers.Adam(1e-3)
    lr_scheduler = ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
    model.compile(optimizer=optimizer, loss=scaled_mse, metrics=['mse'])

    imgs_decompressed = tf.convert_to_tensor(imgs_decompressed)
    imgs = tf.convert_to_tensor(imgs)

    model.fit(imgs_decompressed, imgs, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[lr_scheduler, saver])
    
def test(imgs, imgs_decompressed):
    # load model
    model = build_model()
    model.load_weights('model')
    
    # predict
    imgs_decompressed = tf.convert_to_tensor(imgs_decompressed)
    imgs = tf.convert_to_tensor(imgs)
    preds = model.predict(imgs_decompressed)
    
    # calculate mse with tensorflow
    mse = mean_squared_error(imgs.numpy().flatten(), preds.flatten())
    print(f"MSE: {mse}")
    
train(imgs, imgs_decompressed)
