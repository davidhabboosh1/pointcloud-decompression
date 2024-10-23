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

def pairwise_distance(pc1, pc2):
    pc1_expanded = tf.expand_dims(pc1, axis=1)
    pc2_expanded = tf.expand_dims(pc2, axis=0)

    distances = tf.reduce_sum(tf.square(pc1_expanded - pc2_expanded), axis=-1)

    return distances

def similarity(pc1, pc2):
    pc1 = tf.convert_to_tensor(pc1, dtype=tf.float64)
    pc2 = tf.convert_to_tensor(pc2, dtype=tf.float64)
    
    distances1 = pairwise_distance(pc1, pc2)
    distances2 = pairwise_distance(pc2, pc1)

    mean_distance1 = tf.reduce_mean(tf.reduce_min(distances1, axis=1))
    mean_distance2 = tf.reduce_mean(tf.reduce_min(distances2, axis=1))

    similarity_score = mean_distance1 + mean_distance2

    return -similarity_score

@keras.saving.register_keras_serializable()
def loss(y_true, y_pred):
    return -similarity(tf.cast(y_true, tf.float64), tf.cast(y_pred, tf.float64))


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
    with h5py.File(f'results/{filename}.h5', 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        
    print('finished', flush=True)
    # TODO: Add back in the deletion of the local file
    # os.remove(local_path)
    
def handle_mising_object(file_identifier, sha256, metadata):
    return

@keras.saving.register_keras_serializable()
def sign_activation(x):
    return tf.sign(x)

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
    collect = False
    train = True
    
    if collect:
        if os.path.exists('results'):
            shutil.rmtree('results', ignore_errors=True)
        os.mkdir('results')
        
        for obj in ['liberty', 'cabinet', 'iphone']:
            handle_obj(f'{obj}.obj', None, None, None)
    
    if train:
        # x = []
        # y = []
        # for result in os.listdir('results'):
        #     # read the h5 file
        #     with h5py.File(f'results/{result}', 'r') as f:
        #         for i in range(len(f['x'])):
        #             x.append(f['x'][i])
        #             y.append(f['y'][i])
        #     break #TODO: figure out a a way to get rid of this
        
        # max_len = max([len(i) for i in x])
        # x = pad_sequences(x, maxlen=max_len, dtype='float64')
        # y = pad_sequences(y, maxlen=max_len, dtype='float64')
        
        model = keras.Sequential([
            keras.layers.Input(shape=(None, 3)),  # Input as a 1D sequence
            keras.layers.Reshape((-1, 1, 3)),
            
            # Encoder Block
            keras.layers.ConvLSTM1D(64, 3, padding='same', strides=1, activation='relu'),
            keras.layers.Reshape((-1, 1, 64)),
            keras.layers.ConvLSTM1D(256, 3, padding='same', strides=1, activation='relu'),
            keras.layers.Reshape((-1, 1, 128)),
            
            # Bi-level code
            keras.layers.ConvLSTM1D(32, 1, padding='same', activation='tanh'),  # Approximation for bi-level coding
            keras.layers.Activation(sign_activation),  # Apply binary level conversion (-1, 1)
            
            # Decoder Block (using Conv1DTranspose to upsample)
            keras.layers.Conv1DTranspose(128, 3, padding='same', strides=1, activation='relu'),
            keras.layers.Conv1DTranspose(64, 3, padding='same', strides=1, activation='relu'),
            
            # Output layer
            keras.layers.Conv1D(3, 1, padding='same', activation='linear'),
            keras.layers.Reshape((-1, 3))
        ])
        
        # Define Sequential model
        # model = keras.Sequential([
        #     keras.layers.Input(shape=(None, 3)),  # Input as a 1D sequence
        #     keras.layers.Reshape((-1, 1, 3)),
            
        #     # Encoder Block
        #     keras.layers.ConvLSTM1D(64, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Reshape((-1, 1, 64)),
        #     keras.layers.ConvLSTM1D(256, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Reshape((-1, 1, 256)),
        #     keras.layers.ConvLSTM1D(512, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Reshape((-1, 1, 512)),
        #     keras.layers.ConvLSTM1D(512, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Reshape((-1, 1, 512)),
            
        #     # Bi-level code
        #     keras.layers.ConvLSTM1D(32, 1, padding='same', activation='tanh'),  # Approximation for bi-level coding
        #     keras.layers.Activation(sign_activation),  # Apply binary level conversion (-1, 1)
            
        #     # Decoder Block (using Conv1DTranspose to upsample)
        #     keras.layers.Conv1DTranspose(512, 1, padding='same', activation='relu'),
        #     keras.layers.Conv1DTranspose(512, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Conv1DTranspose(256, 3, padding='same', strides=1, activation='relu'),
        #     keras.layers.Conv1DTranspose(128, 3, padding='same', strides=1, activation='relu'),
            
        #     # Output layer
        #     keras.layers.Conv1D(3, 1, padding='same', activation='linear'),
        #     keras.layers.Reshape((-1, 3))
        # ])
        
        # compile with the loss being the similarity between the two point clouds
        # model.compile(optimizer='adam', loss='mean_squared_error')
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)  # Lower starting learning rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)

        model.compile(optimizer=optimizer, loss=loss)
        # index = np.random.randint(0, len(x))
        # model.fit(np.array([x[index]]), np.array([y[index]]), epochs=1000, callbacks=[reduce_lr])
        
        file_list = [f'results/{i}' for i in os.listdir('results')]
        batch_size = 1
        start_quant = 10
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
                example = model.predict(np.array([example]))[0]
                print(f'Done with {i} quantization')
            
            print(f'On {result}:')
            print(f'Similarity between quantization {start} and uncompressed: {similarity(y[29], x[start]).numpy().item()}')
            print(f'Similarity between quantization 30 and uncompressed: {similarity(y[29], x[29]).numpy().item()}')
            print(f'Similarity between prediction and uncompressed: {similarity(y[29], example).numpy().item()}')
            print()

if __name__ == '__main__':
    main()