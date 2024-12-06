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

class ImgDataset(Dataset):
    def __init__(self, decompressed_imgs, target_imgs, transform=None):
        self.decompressed = decompressed_imgs
        self.target = target_imgs
        self.transform = transform

    def __len__(self):
        return len(self.decompressed)

    def __getitem__(self, idx):
        img_decompressed = self.decompressed[idx]
        img_target = self.target[idx]

        if self.transform:
            img_decompressed = self.transform(img_decompressed)
            img_target = self.transform(img_target)

        # Convert to float32 (if it's not already)
        img_decompressed = img_decompressed.float()
        img_target = img_target.float()

        return img_decompressed, img_target

pcs = np.random.rand(10000, IMG_SIZE ** 2, 3).astype(np.float32)
imgs = pc2img(pcs)
imgs_compressed = [compress(pc, 5, 30) for pc in pcs]
imgs_decompressed = [decompress(img) for img in imgs_compressed]
imgs_decompressed = pc2img(np.array(imgs_decompressed, dtype=np.float32))

print(f"imgs_decompressed shape: {imgs_decompressed.shape}")
print(f"imgs shape: {imgs.shape}")
print(f"Data type of imgs_decompressed: {imgs_decompressed.dtype}")
print(f"Data type of imgs: {imgs.dtype}")

def unet_model(input_shape=(64, 64, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(b)
   
    # Decoder
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c6)

    model = models.Model(inputs, outputs)
    return model

if os.path.exists('unet_model.keras'):
    model = models.load_model('unet_model.keras')
    print('Loaded model from disk with shape:', model.input_shape)
else:
    model = unet_model()
    print('Created new model with shape:', model.input_shape)
   

optimizer = keras.optimizers.Adam(1e-3)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
saver = ModelCheckpoint('unet_model', save_best_only=True, verbose=1, save_format='tf')
model.compile(optimizer=optimizer, loss='mean_squared_error')

imgs_decompressed = tf.convert_to_tensor(imgs_decompressed)
imgs = tf.convert_to_tensor(imgs)

model.fit(imgs_decompressed, imgs, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler, saver])
