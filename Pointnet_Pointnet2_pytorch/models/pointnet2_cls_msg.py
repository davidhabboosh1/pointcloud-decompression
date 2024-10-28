import torch.nn as nn
import torch.nn.functional as F
import torch
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024 + 10, 512) # +10 for the quantization
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        
        # layer for quantization
        self.quant_fc1 = nn.Linear(1, 20)
        self.quant_fc2 = nn.Linear(20, 10)

    def forward(self, xyz, quantization):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        quantization = quantization.view(B, 1)

        # Process quantization level through multiple layers
        quant_features = F.relu(self.quant_fc1(quantization))  # First layer
        quant_features = F.relu(self.quant_fc2(quant_features))  # Second layer

        # Concatenate point cloud features with quantization features
        x = torch.cat((x, quant_features), dim=1)

        # Pass through fully connected layers
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # use mean squared error
        total_loss = F.mse_loss(pred, target)
        
        # total_loss = F.nll_loss(pred, target)

        return total_loss


