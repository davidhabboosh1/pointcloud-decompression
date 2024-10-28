from main2 import compress, decompress, pad_or_trim
import torch
from Pointnet_Pointnet2_pytorch.models import pointnet2_cls_msg
import os
import numpy as np
import tensorflow as tf

def chamfer_distance(pc1, pc2):
    # Ensure the inputs are of type float64
    pc1 = tf.cast(pc1, dtype=tf.float64)
    pc2 = tf.cast(pc2, dtype=tf.float64)

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

min_quant = 10
num_rand = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

classifier = pointnet2_cls_msg.get_model(4096 * 3, normal_channel=False).to(device)
if os.path.exists('classifier.pth'):
    classifier.load_state_dict(torch.load('classifier.pth'))
classifier.eval()

for i in range(num_rand):
    test_obj_orig = np.random.rand(4096, 3)
    
    for quant in range(min_quant, 31):
        compressed = compress(test_obj_orig, 10, quant)
        decompressed = decompress(compressed)
    
        test_obj = np.expand_dims(decompressed, axis=0)
        test_obj = torch.Tensor(test_obj).to(device)
        test_obj = test_obj.transpose(2, 1)
        pred, _ = classifier(test_obj, torch.Tensor([quant]).to(device))
        decompressed = torch.Tensor(decompressed).to(device)
        
        pred = pred.cpu().detach().numpy().reshape(4096, 3)
        decompressed = decompressed.cpu().detach().numpy()
        decompressed = pad_or_trim(decompressed).reshape(4096, 3)
        
        pred_diff = chamfer_distance(decompressed, pred)
        true_diff = chamfer_distance(decompressed, test_obj_orig)
        winner = 'Pred better' if pred_diff < true_diff else 'True better'
        
        print(f'Quantization level: {quant}, Pred Diff: {pred_diff}, True Diff: {true_diff}, {winner}')
        
    print()