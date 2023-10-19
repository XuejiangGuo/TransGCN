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


class ssGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, drop_rate=0.3):
        super(ssGCN, self).__init__()
        self.drop_rate = drop_rate
        self.cov1 = GCNConv(input_dim, hidden_dim1)
        self.cov2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.cov1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.cov2(x, edge_index)
        x = F.dropout(F.relu(x), p=self.drop_rate, training=self.training)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    pass
