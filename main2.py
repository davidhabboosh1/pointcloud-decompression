import objaverse.xl as oxl
import objaverse_xl as oxl2
import os
import shutil
import trimesh
import DracoPy
import h5py
import numpy as np
from scipy.spatial import KDTree
import tempfile
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow_probability as tfp

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
        
    # with FileLock('results.h5.lock'):
    #     with h5py.File('results.h5', 'a') as f:
    #         x_dataset = f['x']
    #         y_dataset = f['y']
    #         for i in range(1, len(results)):
    #             x_dataset.resize(x_dataset.shape[0] + 1, axis=0)
    #             y_dataset.resize(y_dataset.shape[0] + 1, axis=0)
                
    #             x_dataset[-1] = results[i-1]
    #             y_dataset[-1] = results[i]
    
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

def main():
    collect = True
    train = True
    
    if collect:
        if os.path.exists('results'):
            shutil.rmtree('results', ignore_errors=True)
        os.mkdir('results')
        
        # if os.path.exists('obj_temp'):
        #     shutil.rmtree('obj_temp', ignore_errors=True)
        # os.mkdir('obj_temp')
        
        # tempfile.tempdir = 'obj_temp'
        
        # with h5py.File('results.h5', 'w') as f:
        #     f.create_dataset('x', (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        #     f.create_dataset('y', (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        
        # annotations = oxl.get_annotations()
        # annotations = annotations[annotations['source'] != 'thingiverse']
        # oxl.download_objects(annotations, handle_found_object=handle_obj, handle_missing_object=handle_mising_object)
        
        # downloader = oxl2.ObjaverseDownloader()
        # annotations = downloader.get_annotations()
        # downloader.download_objects(annotations, handle_found_object=handle_obj)
        
        # load the h5 file and calculate the similarity between two point clouds at some random index
        # with h5py.File('results.h5', 'r') as f:
        #     x = f['x']
        #     y = f['y']
        #     index = np.random.randint(0, x.shape[0])
        #     print(similarity(x[index], y[index]))
        
        for obj in ['liberty', 'cabinet', 'iphone']:
            handle_obj(f'{obj}.obj', None, None, None)
    
    if train:
        x = []
        y = []
        for result in os.listdir('results'):
            # read the h5 file
            with h5py.File(f'results/{result}', 'r') as f:
                for i in range(len(f['x'])):
                    x.append(f['x'][i])
                    y.append(f['y'][i])
            break # TODO: remove so we process all three
        
        max_len = max([len(i) for i in x])
        x = pad_sequences(x, maxlen=max_len, dtype='float64')
        y = pad_sequences(y, maxlen=max_len, dtype='float64')
        
        # create a basic cnn that overfits to these three objects
        # TODO: maybe use Conv3D instead
        # model = keras.Sequential([
        #     keras.layers.Input(shape=(max_len, 3)),
        #     keras.layers.Conv1D(64, 3, activation='relu'),
        #     keras.layers.MaxPooling1D(2),
        #     keras.layers.Conv1D(256, 3, activation='relu'),
        #     keras.layers.MaxPooling1D(2),
        #     keras.layers.Conv1D(512, 3, activation='relu'),
        #     keras.layers.MaxPooling1D(2),
        #     keras.layers.Conv1D(512, 3, activation='relu'),
        #     keras.layers.MaxPooling1D(2),
        #     keras.layers.Conv1D(32, 1, activation='relu'),
        #     # bi-level code
        #     keras.layers.Conv1DTranspose(32, 1, activation='relu'),
        #     keras.layers.UpSampling1D(2),
        #     keras.layers.Conv1DTranspose(512, 3, activation='relu'),
        #     keras.layers.UpSampling1D(2),
        #     keras.layers.Conv1DTranspose(512, 3, activation='relu'),
        #     keras.layers.UpSampling1D(2),
        #     keras.layers.Conv1DTranspose(256, 3, activation='relu'),
        #     keras.layers.UpSampling1D(2),
        #     keras.layers.Conv1DTranspose(64, 3, activation='relu'),
        #     keras.layers.UpSampling1D(2),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(256, activation='relu'),
        #     keras.layers.Dense(max_len * 3),
        #     keras.layers.Reshape((max_len, 3))
        # ])
        
        model = keras.Sequential([
            # Input image
            keras.layers.Input(shape=(max_len, 3)),
            
            # 64 filter 3x3 feed forward conv
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            
            # 256 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1D(256, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(256, 1, activation='relu'),
            keras.layers.LSTM(256, return_sequences=True),
            
            # 512 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1D(512, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(512, 1, activation='relu'),
            keras.layers.LSTM(512, return_sequences=True),
            
            # 512 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1D(512, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(512, 1, activation='relu'),
            keras.layers.LSTM(512, return_sequences=True),
            
            # 32 filter 1x1 feed forward conv
            keras.layers.Conv1D(32, 1, activation='relu'),
            
            # bi-level code
            
            # 32 filter 1x1 feed forward conv
            keras.layers.Conv1DTranspose(32, 1, activation='relu'),
            
            # 512 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1DTranspose(512, 3, activation='relu'),
            keras.layers.Conv1D(512, 1, activation='relu'),
            keras.layers.LSTM(512, return_sequences=True),
            
            # 512 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1DTranspose(512, 3, activation='relu'),
            keras.layers.Conv1D(512, 1, activation='relu'),
            keras.layers.LSTM(512, return_sequences=True),
            
            # 256 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1DTranspose(256, 3, activation='relu'),
            keras.layers.Conv1D(256, 1, activation='relu'),
            keras.layers.LSTM(256, return_sequences=True),
            
            # 128 filter 3x3 I/P, 1x1 LSTM conv
            keras.layers.Conv1DTranspose(128, 3, activation='relu'),
            keras.layers.Conv1D(128, 1, activation='relu'),
            keras.layers.LSTM(128, return_sequences=True),
            
            # output
            keras.layers.Flatten(),
            keras.layers.Dense(max_len * 3),
            keras.layers.Reshape((max_len, 3))
        ])
        
        # compile with the loss being the similarity between the two point clouds
        # model.compile(optimizer='adam', loss='mean_squared_error')
        optimizer = keras.optimizers.Adam(learning_rate=0.1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(x, y, epochs=100, callbacks=[reduce_lr])
        
        model.save('model.keras')
    
    model = keras.models.load_model('model.keras')
    index = np.random.randint(0, len(x))
    prediction = model.predict(np.expand_dims(x[index], axis=0))
    
    # print similarity between the prediction and the actual
    print(similarity(y[index], prediction[0]).numpy().item())

if __name__ == '__main__':
    main()