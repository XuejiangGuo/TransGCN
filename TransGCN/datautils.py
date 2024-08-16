# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : datautils.py
# @Time     : 7/10/2023 10:54 PM
# @Emal     : wangbing587@163.com
# @Desc     :


from utils import *
from RFclf import *
from MNN import *
from DistanceFeature import *
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np


class DataProcess:
    def __init__(self, subcount=500, p=0.95, c=10, k=1, val_size=0.2, seed=0):
        self.subcount = subcount
        self.seed = seed
        self.p = p
        self.c = c
        self.k = k
        self.val_size = val_size

    def train(self,  rawdata, R):
        print('++++++++++++++++++++Data Processing+++++++++++++++++++++++')
        self.r = R
        self.an_col = generate_an_col(rawdata, R)
        self.df_real = data_norma(rawdata, R)
        self.df_syn = generate_synthetic_data(self.df_real, self.an_col, self.p, self.c)

        self.markers_map, self.markers_revmap = label_mapping(self.df_real.loc[self.df_real['markers'] != 'unknown', 'markers'])
        self.markersSYN_map, self.markersSYN_revmap = label_mapping(self.df_syn['markers_SYN'])
        self.df_syn['markers_SYN'] = self.df_syn['markers_SYN'].map(self.markersSYN_map)

        mk = np.array(list(self.markersSYN_map.keys()))
        self.S_index = torch.tensor(np.argwhere(mk[:, 0] == mk[:, 1]).flatten())

        self.PTF_map, self.PTF_revmap = label_mapping(self.df_syn['PTF'])
        self.df_syn['PTF'] = self.df_syn['PTF'].map(self.PTF_map)

        self.df_synsub = subsample(self.df_syn, label_column='markers_SYN', subcount=self.subcount)
        self.df_syn = self.df_syn.sample(min(int(1e6), self.df_syn.shape[0]))
        self.y1synsub_label = torch.tensor(self.df_synsub['markers_SYN'])
        self.y2synsub_label = torch.tensor(self.df_synsub['PTF'])

        self.y1real_rfprob, self.y1val_RFacc = RFclf(x=self.df_syn[self.an_col['Sample_id']],
                                                    y=self.df_syn['markers_SYN'],
                                                    x_test=self.df_real[self.an_col['Sample_id']],
                                                   val_size=self.val_size, seed=self.seed)
        self.y2real_rfprob, self.y2val_RFacc = RFclf(x=self.df_syn[self.an_col['Sample_id']],
                                                     y=self.df_syn['PTF'],
                                                     x_test=self.df_real[self.an_col['Sample_id']],
                                                     val_size=self.val_size, seed=self.seed)
        self.y1real_rfprob, self.y2real_rfprob = torch.tensor(self.y1real_rfprob), torch.tensor(self.y2real_rfprob)

        x_synsub = self.df_synsub[self.an_col['Sample_id']]
        x_real = self.df_real[self.an_col['Sample_id']]
        feature_exp = pd.concat([x_synsub, x_real], axis=0)
        feature_dis = DistanceFeature(feature_exp, R)
        self.feature = torch.tensor(StandardScaler().fit_transform(np.hstack((feature_exp.values, feature_dis))),
                                    dtype=torch.float32)
        self.edge = MNN_Hybrid(x_synsub, x_real, self.k).coalesce()

        train_idx, val_idx = data_split(self.df_synsub, label_column='markers_SYN', val_size=self.val_size)
        self.train_mask, self.val_mask = torch.tensor(train_idx), torch.tensor(val_idx)
        self.real_mask = torch.arange(self.df_synsub.shape[0], self.feature.shape[0])


if __name__ == '__main__':
    pass
