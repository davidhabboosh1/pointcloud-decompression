from main2 import compress, decompress, pad_or_trim
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
        pred, _ = classifier(test_obj, torch.Tensor([quant]).to(device))
        decompressed = torch.Tensor(decompressed).to(device)
        
        pred = pred.cpu().detach().numpy().reshape(4096, 3)
        decompressed = decompressed.cpu().detach().numpy()
        decompressed = pad_or_trim(decompressed).reshape(4096, 3)
        test_obj = test_obj.cpu().detach().numpy().reshape(4096, 3)
        
        pred_diff = np.linalg.norm(decompressed - pred)
        true_diff = np.linalg.norm(decompressed - test_obj)
        winner = 'Pred better' if pred_diff < true_diff else 'True better'
        print(f'Quantization level: {quant}, Pred Diff: {pred_diff}, True Diff: {true_diff}, {winner}')
        
    print()