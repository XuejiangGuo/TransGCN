# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : utils.py
# @Time     : 5/16/2023 12:27 PM
# @Emal     : wangbing587@163.com
# @Desc     :

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from Zscore import Zscoretest


def generate_an_col(rawdata, R):
    F = int(len(rawdata.columns[:-1]) / R / 2)
    an_col = pd.DataFrame({'Raw_id': rawdata.columns[:-1]})
    an_col['Sample_id'] = [f'{x}_R{y}_F{z}' for x in ['T1', 'T2'] for y in range(1, R + 1) for z in range(1, F + 1)]
    an_col['Type'] = [i.split('_')[0] for i in an_col['Sample_id']]
    an_col['Repeat'] = [i.split('_')[1] for i in an_col['Sample_id']]
    an_col['Fraction'] = [i.split('_')[2] for i in an_col['Sample_id']]
    return an_col


def data_norma(rawdata, R):
    an_col = generate_an_col(rawdata, R)
    F = int(an_col.shape[0] / R / 2)
    exprs = rawdata[an_col['Raw_id']].values
    data = pd.DataFrame(data=np.hstack([exprs[:, i * F:(i + 1) * F] / exprs[:, i * F:(i + 1) * F].sum(1).reshape(-1, 1)
                                        for i in range(R * 2)]), columns=an_col['Sample_id'].tolist(), index=rawdata.index)
    data['ProteinID'] = rawdata.index
    data['markers'] = rawdata['markers']
    data = data.dropna().round(3)
    return data


def generate_synthetic_data(data, an_col, p=0.95, c=10):
    df0 = deepcopy(data[data['markers'] != 'unknown'])
    ZS = Zscoretest()
    ZS.fit(df0[an_col['Sample_id']], df0['markers'])
    df0['ZSpvalue'] = ZS.pvalue(df0[an_col['Sample_id']], df0['markers'])
    df0['IntraRank'] = df0.groupby('markers')['ZSpvalue'].rank()
    df = df0[(df0['ZSpvalue'] < p) | (df0['IntraRank'] <= c)]
    df_T1 = df[an_col.loc[an_col['Type'] == 'T1', 'Sample_id']]
    df_T2 = df[an_col.loc[an_col['Type'] == 'T2', 'Sample_id']]
    df_syn = pd.concat([pd.DataFrame(data=np.array([np.hstack([i, j]) for i in df_T1.values for j in df_T2.values]),
                                     columns=an_col['Sample_id'].tolist()),
                        pd.DataFrame([(i, j) for i in df.index for j in df.index],
                                     columns=['ProteinID_T1', 'ProteinID_T2']),
                        pd.DataFrame([(i, j) for i in df['markers'] for j in df['markers']],
                                     columns=['markers_T1', 'markers_T2'])], axis=1)
    df_syn['ProteinID_SYN'] = list(zip(df_syn['ProteinID_T1'], df_syn['ProteinID_T2']))
    df_syn['markers_SYN'] = list(zip(df_syn['markers_T1'], df_syn['markers_T2']))
    df_syn.index = df_syn['ProteinID_SYN']
    df_syn['PTF'] = 'Transport'
    df_syn.loc[df_syn['markers_T1'] == df_syn['markers_T2'], 'PTF'] = 'Stable'
    df_syn = df_syn[df_syn['ProteinID_T1'] != df_syn['ProteinID_T2']]
    return df_syn


def label_mapping(y):
    label_map = dict(zip(sorted(set(y)), range(len(set(y)))))
    label_revmap = dict(zip(range(len(set(y))), sorted(set(y))))
    return label_map, label_revmap


def subsample(df, label_column, subcount=500, seed=0):
    df_sub = pd.DataFrame()
    for i in sorted(df[label_column].unique()):
        df_sub_i = df[df[label_column] == i]
        df_sub = pd.concat([df_sub, df_sub_i.sample(min(subcount, df_sub_i.shape[0]), random_state=seed)], axis=0)
    return df_sub


def data_split(data, label_column, val_size,  seed=0):
    data = deepcopy(data)
    data.index = range(data.shape[0])
    x_train, x_val, y_train, y_val = train_test_split(data, data[label_column], stratify=data[label_column],
                                                      test_size=val_size, random_state=seed)
    return x_train.index.tolist(), x_val.index.tolist()





