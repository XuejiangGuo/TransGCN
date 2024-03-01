# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : Zscore.py
# @Time     : 5/25/2023 9:44 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import numpy as np
import pandas as pd
from scipy.stats import norm


class Zscoretest:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.X_mean = {}
        self.X_std = {}
        for i in sorted(np.unique(y)):
            Xi = X[np.argwhere(y == i).flatten()]
            self.X_mean[i] = Xi.mean(axis=0)
            self.X_std[i] = Xi.std(axis=0)

    def pvalue(self, X_test, y_test):
        return 1 - 2 * norm.sf(np.abs(self.zsore(X_test, y_test)).max(axis=1))

    def zsore(self, X_test, y_test):
        X_test, y_test = np.array(X_test), np.array(y_test)
        self.X_zscore = np.zeros_like(X_test)
        for i in sorted(np.unique(y_test)):
            idx = np.argwhere(y_test == i).flatten()
            Xi = X_test[idx]
            self.X_zscore[idx] = (Xi - self.X_mean[i]) / self.X_std[i]
        return self.X_zscore


if __name__ == '__main__':
    pass

