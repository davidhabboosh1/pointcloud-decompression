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

class SAMPLER(layers.Layer):
    def __init__(self, npoints):
        super(SAMPLER, self).__init__()
        self.npoints = npoints

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_points = tf.shape(inputs)[1]
        indices = tf.random.uniform((batch_size, self.npoints), minval=0, maxval=num_points, dtype=tf.int32)
        return tf.gather(inputs, indices, batch_dims=1)

class GROUPER(layers.Layer):
    def __init__(self, npoints, nsample):
        super(GROUPER, self).__init__()
        self.npoints = npoints
        self.nsample = nsample
        self.sampler = SAMPLER(npoints)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_points = tf.shape(inputs)[1]

        # Sample centroids
        centroids = self.sampler(inputs)
        
        # Generate random indices for grouping
        group_indices = tf.random.uniform((batch_size, self.npoints, self.nsample), minval=0, maxval=num_points, dtype=tf.int32)
        grouped_points = tf.gather(inputs, group_indices, batch_dims=1)
        return grouped_points

class PointNetLayer(layers.Layer):
    def __init__(self, npoints, nsample):
        super(PointNetLayer, self).__init__()
        self.sampler = SAMPLER(npoints)
        self.grouper = GROUPER(npoints, nsample)

    def call(self, inputs):
        sampled_points = self.sampler(inputs)
        grouped_points = self.grouper(inputs)
        return tf.reduce_max(grouped_points, axis=2)  # Use max pooling for simplicity

class PointNetPlusPlus(models.Model):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.layer1 = PointNetLayer(npoints=512, nsample=32)
        self.layer2 = PointNetLayer(npoints=128, nsample=32)
        self.mlp1 = layers.Dense(256, activation='relu')
        self.mlp2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return self.output_layer(x)

# compress a mesh file using DracoPy
def compress(mesh, compression_level=0, quantization=14):
    original = DracoPy.DracoPointCloud({'encoding_options_set': False, 'points': mesh.vertices.flatten('C')})
    if quantization == 0:
        compressed = DracoPy.encode(mesh.vertices.flatten('C'), compression_level=compression_level)
    else:
        compressed = DracoPy.encode(mesh.vertices.flatten('C'), compression_level=compression_level, quantization_bits=quantization)
    
    return original, compressed

# decompress a compressed mesh file using DracoPy
def decompress(compressed):
    decompressed = DracoPy.decode(compressed)
    
    return decompressed

def chamfer_distance(pc1, pc2):
    # Compute the distance from pc1 to pc2
    pc1_expanded = tf.expand_dims(pc1, axis=1)
    pc2_expanded = tf.expand_dims(pc2, axis=0)

    distances = tf.reduce_sum(tf.square(pc1_expanded - pc2_expanded), axis=-1)

    # Get the minimum distances in each direction
    min_distances1 = tf.reduce_mean(tf.reduce_min(distances, axis=1))
    min_distances2 = tf.reduce_mean(tf.reduce_min(distances, axis=0))

    # Chamfer Distance is the sum of both directions
    chamfer_dist = min_distances1 + min_distances2

    return chamfer_dist

@keras.saving.register_keras_serializable()
def loss(y_true, y_pred):
    return chamfer_distance(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))

def normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    diag_norm = np.max(np.linalg.norm(points, axis=1))
    points = points / diag_norm
    return points, centroid, diag_norm

def handle_obj(local_path, file_identifier, sha256, metadata):
    ext = os.path.splitext(local_path)[1]
    if ext == '.blend':
        return
    
    mesh = trimesh.load(local_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    
    results = []
    for i in list(range(1, 31)) + [0]:
        _, compressed = compress(mesh, 10, i)
        points = decompress(compressed).points
        
        results.append(points)
    
    filename = os.path.basename(local_path).split('.')[0]
    print(filename, flush=True)
    
    x = np.array(results[:-1], dtype=np.float64)
    y = np.array(results[1:], dtype=np.float64)
    
    # normalize and downsample/pad
    x, _, _ = normalize(x)
    y, _, _ = normalize(y)
    
    with h5py.File(f'results/{filename}.h5', 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        
    print('finished', flush=True)
    # TODO: Add back in the deletion of the local file
    # os.remove(local_path)
    
def handle_mising_object(file_identifier, sha256, metadata):
    return

def data_generator(file_list, batch_size=1, start=0):
    while True:
        for file_name in file_list:
            with h5py.File(file_name, 'r') as f:
                x_data = f['x'][:]
                y_data = f['y'][:]
                num_samples = len(x_data)

                for i in range(start, num_samples, batch_size):
                    x_batch = x_data[i:i + batch_size]
                    y_batch = y_data[i:i + batch_size]
                    yield x_batch, y_batch

def main():
    collect = True
    train = True
    
    if collect:
        if os.path.exists('results'):
            shutil.rmtree('results', ignore_errors=True)
        os.mkdir('results')
        
        for obj in ['liberty', 'cabinet', 'iphone']:
            handle_obj(f'{obj}.obj', None, None, None)
            
        for result in os.listdir('results'): # find the 
            with h5py.File(f'results/{result}', 'r') as f:
                print(f'{result}: {len(f["x"])}')
    
    if train:
        # model = keras.Sequential([
        #     keras.layers.Input(shape=(4096,3)),

        #     # Encoder
        #     keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
        #     keras.layers.MaxPooling1D(pool_size=2),

        #     keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
        #     keras.layers.MaxPooling1D(pool_size=2),

        #     keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
        #     keras.layers.MaxPooling1D(pool_size=2),

        #     # Bottleneck
        #     keras.layers.Conv1D(512, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(512, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),

        #     # Decoder
        #     keras.layers.UpSampling1D(size=2),
        #     keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),

        #     keras.layers.UpSampling1D(size=2),
        #     keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),

        #     keras.layers.UpSampling1D(size=2),
        #     keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),
            
        #     keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.LeakyReLU(alpha=0.1),

        #     # Output Layer
        #     keras.layers.Conv1D(3, kernel_size=1, padding='same', activation='linear')
        # ])
        
        model = PointNetPlusPlus(4096 * 3)
        
        # compile with the loss being the similarity between the two point clouds
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.5)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-4)

        model.compile(optimizer=optimizer, loss=loss)
        
        file_list = [f'results/{i}' for i in os.listdir('results')] # only use first for now
        
        batch_size = 1
        start_quant = 20
        steps_per_epoch = (30 // batch_size - start_quant) * len(file_list)
        model.fit(data_generator(file_list, batch_size, start_quant),
                  steps_per_epoch=steps_per_epoch, epochs=100, 
                  callbacks=[reduce_lr],
                  shuffle=True
        )
        
        model.save('model.keras')
    
    model = keras.models.load_model('model.keras')
    
    for result in os.listdir('results'):
        with h5py.File(f'results/{result}', 'r') as f:
            # start is the quantization level of the x we want, go to 31
            start = 12
            x = f['x']
            y = f['y']
            example = x[start]
            
            for i in range(start, 30):
                # example, centroid, diag_norm = normalize(example)
                # example_unnormed = example * diag_norm + centroid
                # print('Similarity between norm and orig,', similarity(example_unnormed, example).numpy().item()) 
                prediction = model.predict(example)
                example = prediction.reshape((4096, 3))
                # example = example * diag_norm + centroid
                print(f'Done with {i} quantization')
            
            print(f'On {result}:')
            print(f'Similarity between quantization {start} and uncompressed: {similarity(y[29], x[start]).numpy().item()}')
            print(f'Similarity between quantization 30 and uncompressed: {similarity(y[29], x[29]).numpy().item()}')
            print(f'Similarity between prediction and uncompressed: {similarity(y[29], example).numpy().item()}')
            print()

if __name__ == '__main__':
    main()
