# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : MNN.py
# @Time     : 5/15/2023 10:12 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def MNN_Inter(data1, data2, k=1):
    neigh1 = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(data2)
    distances1, indices1 = neigh1.kneighbors(data1, n_neighbors=k)
    indices1 = np.insert(indices1, 0, range(data1.shape[0]), axis=1)
    indices1 = np.vstack([indices1[:, [0, i]] for i in range(1, indices1.shape[1])])
    neigh2 = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(data1)
    distances2, indices2 = neigh2.kneighbors(data2, n_neighbors=k)
    indices2 = np.insert(indices2, 0, range(data2.shape[0]), axis=1)
    indices2 = np.vstack([indices2[:, [0, i]] for i in range(1, indices2.shape[1])])[:, [1, 0]]
    edge = pd.Series([f'{i}_{j}' for i, j in np.vstack([indices1, indices2])]).value_counts()
    idx = np.array([i.split('_') for i in edge[edge == 2].index], dtype=np.int64).T
    return torch.sparse.FloatTensor(indices=torch.tensor(idx, dtype=torch.int64),
                                    values=torch.tensor([1.0] * idx.shape[1],dtype=torch.float32),
                                    size=torch.Size((data1.shape[0], data2.shape[0]))).coalesce()


def MNN_Intra(data, k=1):
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(data)
    distances, indices = neigh.kneighbors(data, n_neighbors=k + 1)
    indices1 = np.vstack([indices[:, [0, i]] for i in range(1, indices.shape[1])])
    edge = pd.Series([f'{i}_{j}' for i, j in np.sort(indices1, axis=1)]).value_counts()
    idx = np.array([i.split('_') for i in edge[edge == 2].index], dtype=np.int64).T
    return torch.sparse.FloatTensor(indices=torch.tensor(idx, dtype=torch.int64),
                                    values=torch.tensor([1.0] * idx.shape[1], dtype=torch.float32),
                                    size=torch.Size((data.shape[0], data.shape[0]))).coalesce()


def MNN_Hybrid(data1, data2, k=1):
    Asr = MNN_Inter(data1, data2, k)
    ns = Asr.size()[0]
    Ass = torch.sparse.FloatTensor(indices=torch.empty((2, 0), dtype=torch.long),
                               values=torch.empty(0),
                               size=torch.Size([ns, ns])).coalesce()
    Arr = MNN_Intra(data2, k)
    return torch.cat((torch.cat((Ass, Asr), dim=1),
                      torch.cat((Asr.transpose(0, 1), Arr), dim=1)), dim=0).coalesce()


if __name__ == '__main__':
    pass
