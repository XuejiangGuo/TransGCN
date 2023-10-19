# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : FDR.py
# @Time     : 10/13/2023 11:23 PM
# @Emal     : wangbing587@163.com
# @Desc     :

import pandas as pd
import numpy as np


def FDR(result, score_column, label_column):
    result = result.sort_values(score_column, ascending=False, axis=0)
    result['relabel'] = 0
    result.loc[result[label_column] != 'unknown', 'relabel'] = 1
    result['PTFfdr'] = np.cumsum(result['relabel']) / np.cumsum(result['relabel']).max()
    return result.drop('relabel', axis=1)
