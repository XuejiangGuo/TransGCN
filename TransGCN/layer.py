# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : layer.py
# @Time     : 5/16/2023 2:41 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SELayer(nn.Module):
    def __init__(self, feature_dim, reduction=4):
        super(SELayer, self).__init__()
        self.U = nn.Linear(feature_dim, feature_dim)

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.U(x)
        w = self.fc(x)
        return x * w, w


class ssGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, drop_rate=0.3):
        super(ssGCN, self).__init__()
        self.drop_rate = drop_rate
        self.att_se = SELayer(input_dim)
        self.cov1 = GCNConv(input_dim, hidden_dim1)
        self.cov2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index):
        x, w = self.att_se(x)
        x = F.relu(self.cov1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.cov2(x, edge_index)
        x = F.dropout(F.relu(x), p=self.drop_rate, training=self.training)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x, w


if __name__ == '__main__':
    pass
