from main2 import compress, decompress
import torch
from Pointnet_Pointnet2_pytorch.models import pointnet2_cls_msg
import os
import numpy as np

min_quant = 10
num_rand = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

classifier = pointnet2_cls_msg.get_model(4096 * 3, normal_channel=False).to(device)
if os.path.exists('classifier.pth'):
    classifier.load_state_dict(torch.load('classifier.pth'))
classifier.eval()

for i in range(num_rand):
    test_obj = np.random.rand(4096, 3)
    
    for quant in range(min_quant, 31):
        compressed = compress(test_obj, 10, quant)
        decompressed = decompress(compressed)
    
        test_obj = np.expand_dims(decompressed, axis=0)
        test_obj = torch.Tensor(test_obj).to(device)
        test_obj = test_obj.transpose(2, 1)
        pred = classifier(test_obj, torch.Tensor([quant]).to(device))[1].cpu().detach().numpy()
        decompressed = torch.Tensor(decompressed).to(device)
        test_obj = test_obj.transpose(2, 1)
        print(f'Quantization level: {quant}, Prediction: {pred}, Pred Diff: {np.linalg.norm(decompressed - pred)}, True Diff: {np.linalg.norm(decompressed - test_obj)}')
        
    print()