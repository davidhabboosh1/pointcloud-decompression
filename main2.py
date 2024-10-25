# import objaverse.xl as oxl
# import objaverse_xl as oxl2
import os
import shutil
import trimesh
import DracoPy
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial import KDTree
from keras import layers, models
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, Dropout, MaxPooling1D, UpSampling1D, Add, Attention
from keras.models import Model
from Pointnet_Pointnet2_pytorch.models import pointnet2_cls_msg
from Pointnet_Pointnet2_pytorch import provider
import torch
import logging
import tqdm

# compress a mesh file using DracoPy
def compress(points, compression_level=0, quantization=14):
    points = points.flatten()
    if quantization == 0:
        compressed = DracoPy.encode(points, compression_level=compression_level)
    else:
        compressed = DracoPy.encode(points, compression_level=compression_level, quantization_bits=quantization)
    
    return compressed

# decompress a compressed mesh file using DracoPy
def decompress(compressed):
    decompressed = DracoPy.decode(compressed).points
    
    return decompressed

def pad_or_trim(points, target_shape=(4096, 3)):
    if points.shape[0] < target_shape[0]:
        # Pad with zeros
        padding = np.zeros((target_shape[0] - points.shape[0], 3))
        return np.vstack([points, padding])
    else:
        # Trim to the target shape
        return points[:target_shape[0]]
        
def data_generator(min_quant=0, batch_size=32):
    while True:
        x_batch, y_batch = [], []
        for _ in range(batch_size):
            rand = np.random.rand(4096, 3)
            
            choices = list(range(min_quant, 31))
            if min_quant > 0:
                choices.append(0)
            
            quantization = np.random.choice(choices)
            compressed = compress(rand, 10, quantization)
            x = pad_or_trim(decompress(compressed))
            
            if quantization > 0:
                compressed = compress(rand, 10, quantization + 1 if quantization < 30 else 0)
                y = pad_or_trim(decompress(compressed))
            else:
                y = rand
            
            # y = y - x + 0.5
            # y = np.clip(y, 0, 1)
            # x = np.clip(x, -0.5, 0.5)
            # y = np.clip(y, -0.5, 0.5)
            # y = y - x + 0.5
            
            x_batch.append(x)
            y_batch.append(y.flatten())

        # convert to pytorch tensors
        # yield torch.Tensor(x_batch), torch.Tensor(y_batch) 
        yield np.array(x_batch), torch.Tensor(np.array(y_batch))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    classifier = pointnet2_cls_msg.get_model(4096 * 3, normal_channel=False).to(device)
    criterion = pointnet2_cls_msg.get_loss()
    classifier.apply(inplace_relu)
    
    optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    steps_per_epoch = 100
    batch_size = 16
    min_quant = 30

    logger = logging.Logger('whatever')

    def log_string(str):
        logger.info(str)
        print(str)

    logger.info('Start training...')
    for epoch in range(0, 100):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 100))
        classifier = classifier.train()
        
        generator = data_generator(min_quant=min_quant, batch_size=batch_size)
        for batch_id in range(steps_per_epoch):
            points, target = next(generator)
            
            optimizer.zero_grad()

            # points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points).to(device)
            points = points.transpose(2, 1)
            target = target.to(device)

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target, trans_feat)

            loss.backward()
            optimizer.step()
            
            log_string(f'Loss on batch {batch_id + 1}/{steps_per_epoch}: {loss.item()}')
            
            global_step += 1

        scheduler.step()
        log_string(f'Loss on epoch {epoch + 1}: {loss.item()}')

    logger.info('End of training...')

if __name__ == '__main__':
    main()