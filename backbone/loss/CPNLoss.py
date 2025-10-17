import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CPNLoss(nn.Module):
    def __init__(self, num_classes, num_centers, feat_dim, init):
        super(CPNLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)


    def forward(self, features):

        center = self.centers

        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)

        c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
        dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)

        dist = dist / float(features.shape[1])

        return -dist
