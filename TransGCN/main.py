# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : main.py
# @Time     : 5/16/2023 2:41 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import argparse
import pandas as pd
import numpy as np
from datautils import DataProcess
from model import TransGCN
import warnings
warnings.filterwarnings('ignore')


def main(file, r):
    rawdata = pd.read_csv(file, index_col=0)
    DP = DataProcess()
    DP.train(rawdata, r)
    TG = TransGCN()
    TG.train(DP)
    an_col = DP.an_col
    result = TG.result
    # an_col.to_csv('an_col.csv', index=None)
    result.to_csv('{}_TransGCN.csv'.format(file.split('.csv')[0]), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TransGCN: a semi-supervised graph convolution network-based framework\
         to infer protein translocations in spatio-temporal proteomics")
    parser.add_argument("-f",
                        "--file",
                        dest='f',
                        type=str,
                        help="file path, first column must be ProteinID, last column must be markers")
    parser.add_argument("-r",
                        "--rep",
                        dest='r',
                        type=int,
                        help="number of repeated paird experiments")
    args = parser.parse_args()
    main(args.f, args.r)




