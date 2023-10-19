# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : DistanceFeature.py
# @Time     : 5/17/2023 2:14 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.spatial.distance import pdist
from scipy import stats
from copy import deepcopy
from joblib import Parallel, delayed


def Distance(args):
    p = deepcopy(args[0])
    q = deepcopy(args[1])
    if args[3]:
        p[p <= 0] = 0.001
        q[q <= 0] = 0.001
    return [pdist(np.vstack([p[i], q[i]]), metric=args[2])[0]
            for i in range(q.shape[0])]


def JS(u, v):
    M = (u + v) / 2
    return 0.5*np.sum(u*np.log(u/M))+0.5*np.sum(v*np.log(v/M))


def DistanceCalc(x, y):
    # 直接距离 9
    A1 = [x, y, 'cityblock', False]  # l1 曼哈顿距离
    A2 = [x, y, 'euclidean', False]  # l2 欧几里得距离
    A3 = [x, y, 'cosine', False]  # cosine 余弦距离
    A4 = [x, y, 'correlation', False]  # pcc 皮尔逊相关系数
    A5 = [x, y, 'chebyshev', False]  # max 切比雪夫距离
    A6 = [x, y, 'canberra', False]  # 兰氏距离 （加权曼哈顿凝距离）
    cov = np.cov((x - y).T)
    inv = np.linalg.inv(cov)
    A7 = [x, y, lambda u, v: np.dot(np.dot(u - v, inv), (u - v).T), False]  # 马氏距离
    A8 = [x, y, lambda u, v: np.abs(np.log2(u / v)).sum(), True]  # Sum(LogFC)
    A9 = [x, y, lambda u, v: np.abs(np.log2(u / v)).max(), True]  # =Max(LogFC)

    # 分布距离 5
    B1 = [x, y, lambda u, v: -np.log(np.sum(np.sqrt(u * v))), True]  # BC 巴氏距离
    B2 = [x, y, lambda u, v: np.sqrt(((np.sqrt(u) - np.sqrt(v)) ** 2).sum()), True]  # 海林格距离
    B3 = [x, y, lambda u, v: -np.sum(u * np.log2(v)), True]  # 交叉熵
    B4 = [x, y, lambda u, v: np.sum(u * np.log2(u / v)), True]  # KL
    B5 = [x, y, JS, True]  # JS

    # 排序距离 6
    C1 = [x, y, lambda u, v: stats.wilcoxon(u - v)[1], False]  # 秩和检验
    C2 = [x, y, lambda u, v: stats.spearmanr(u, v)[0], False]  # 斯皮尔
    C3 = [x, y, lambda u, v: stats.kendalltau(u, v)[0], False]  # 肯德尔秩相关
    C4 = [x, y, lambda u, v: (np.argsort(np.argsort(u)) != np.argsort(np.argsort(v))).sum(), False]  # 汉明距离
    C5 = [x, y, lambda u, v: np.abs((np.argsort(np.argsort(u)) - np.argsort(np.argsort(v)))).sum(), False]
    C6 = [x, y, lambda u, v: np.abs((np.argsort(np.argsort(u)) - np.argsort(np.argsort(v)))).max(), False]
    params = [A1, A2, A3, A4, A5, A6, A7, A8, A9, B1, B2, B3, B4, B5, C1, C2, C3, C4, C5, C6]
    return np.array(Parallel(n_jobs=-1)(delayed(Distance)(args) for args in params)).T


def DistanceFeature(data, R):
    data = np.array(data)
    F = int(data.shape[1] / R / 2)
    df_split = [data[:, i * F:(i + 1) * F] for i in range(R * 2)]
    D = np.hstack([DistanceCalc(df_split[i], df_split[i + R]) for i in range(R)])
    return KNNImputer(n_neighbors=3).fit_transform(D)


if __name__ == '__main__':
    pass



