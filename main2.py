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

# compress a mesh file using DracoPy
def compress(points, compression_level=0, quantization=14):
    points = points.copy()
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
    if points.shape[0] == target_shape[0]:
        return points
    elif points.shape[0] < target_shape[0]:
        # Pad with zeros
        padding = np.zeros((target_shape[0] - points.shape[0], 3))
        return np.vstack([points, padding])
    else:
        # Trim to the target shape
        indices = np.random.choice(points.shape[0], target_shape[0], replace=False)
        indices.sort()
        return points[indices]
   
def normalize_points(rand):
    # Center the points
    centered = rand - np.mean(rand, axis=0)
    
    # Calculate the maximum distance from the center
    max_distance = np.max(np.linalg.norm(centered, axis=1))
    
    if max_distance == 0:
        raise ValueError('Max distance is zero, cannot normalize.')
    
    # Normalize to unit sphere
    normalized = centered / max_distance
    
    # Scale to [0, 1] by shifting and scaling
    # Assuming normalized points are in the range [-1, 1]
    scaled = (normalized + 1) / 2  # Shift to [0, 2] and then scale to [0, 1]
    
    # Truncate or pad to 4096 points
    scaled = pad_or_trim(scaled)
    
    # Error if any points are < 0 or > 1
    if np.any(scaled < 0) or np.any(scaled > 1):
        print(scaled)
        raise ValueError('Points are not normalized to [0, 1].')
    
    if scaled.shape != (4096, 3):
        raise ValueError('Points are not of shape (4096, 3).')
    
    return scaled
        
def pad_normalize(points):
    points = pad_or_trim(points)
    points = normalize_points(points)
    
    return points
        
def data_generator(min_quant=0, batch_size=32, random=True):
    while True:
        x_batch, y_batch, quantizations = [], [], []
        for _ in range(batch_size):
            if random:
                # randomly generate 4096 points
                rand = np.random.rand(4096, 3)
            else:
                # randomly choose from cabinet, iphone, and liberty.obj
                rand = trimesh.load(np.random.choice(['cabinet.obj', 'iphone.obj', 'liberty.obj']))
                if isinstance(rand, trimesh.Scene):
                    rand = rand.to_mesh()
                rand = rand.vertices
            
            rand = pad_normalize(rand)
            
            choices = list(range(min_quant, 31))
            if min_quant > 0:
                choices.append(0)
            
            quantization = np.random.choice(choices)
            quantizations.append(quantization)
            
            compressed = compress(rand, 10, quantization)
            x = decompress(compressed)
            x = pad_or_trim(x)
            # x = pad_normalize(x)
            
            # if quantization > 0:
            #     compressed = compress(rand, 10, quantization + 1 if quantization < 30 else 0)
            #     y = decompress(compressed)
            #     y = pad_or_trim(y)
            #     # y = pad_normalize(y)
            # else:
            #     y = rand
            y = rand
            
            x_batch.append(x)
            y_batch.append(y.flatten())

        # convert to pytorch tensors
        # yield torch.Tensor(x_batch), torch.Tensor(y_batch) 
        yield np.array(x_batch), torch.Tensor(np.array(y_batch)), np.array(quantizations)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main():
    # create loss file
    if os.path.exists('loss.txt'):
        with open('loss.txt', 'r') as f:
            if os.stat('loss.txt').st_size == 0:
                min_loss = np.inf
            else:
                min_loss = min(float(line) for line in f)
        os.remove('loss.txt')
    else:
        min_loss = np.inf
    open('loss.txt', 'w').close()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    classifier = pointnet2_cls_msg.get_model(4096 * 3, normal_channel=False).to(device)
    if os.path.exists('classifier.pth'):
        classifier.load_state_dict(torch.load('classifier.pth'))
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
    global_step = 0
    steps_per_epoch = 100
    batch_size = 16
    min_quant = 10
    epochs = 100
    random = True

    logger = logging.Logger('whatever')

    def log_string(str):
        logger.info(str)
        print(str)

    logger.info('Start training...')
    for epoch in range(0, epochs):
        epoch_loss = 0
        
        log_string('Epoch %d/%s:' % (epoch + 1, epochs))
        classifier = classifier.train()
        
        generator = data_generator(min_quant=min_quant, batch_size=batch_size, random=random)
        for batch_id in range(steps_per_epoch):
            points, target, quantizations = next(generator)
            
            optimizer.zero_grad()

            # points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points).to(device)
            points = points.transpose(2, 1)
            target = target.to(device)
            
            quantizations = torch.Tensor(quantizations).to(device)

            pred, trans_feat = classifier(points, quantizations)
            loss = criterion(pred, target, trans_feat)

            loss.backward()
            optimizer.step()
            
            log_string(f'Loss on batch {batch_id + 1}/{steps_per_epoch}: {loss.item()}')
            epoch_loss += loss.item()
            
            global_step += 1

        scheduler.step()
        log_string(f'Loss on epoch {epoch + 1}: {loss.item()}')
        
        epoch_loss /= steps_per_epoch
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(classifier.state_dict(), 'classifier.pth')
            print(f'Model saved with loss {epoch_loss}.')
        with open('loss.txt', 'a') as f:
            f.write(f'{epoch_loss}\n')

    logger.info('End of training...')
    torch.save(classifier.state_dict(), 'classifier.pth')

if __name__ == '__main__':
    main()