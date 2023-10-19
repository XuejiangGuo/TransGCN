# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : loss_function.py
# @Time     : 5/11/2023 9:54 PM
# @Emal     : wangbing587@163.com
# @Desc     :

import torch
import torch.nn as nn


def loss_func(y_label_pred, y_unlabel_pred, y_label, y_unlabel_rfprob,
              alpha):
    loss1 = nn.NLLLoss()(y_label_pred, y_label)
    loss2 = -(y_unlabel_pred * y_unlabel_rfprob).sum(1).mean()
    return loss1 + alpha * loss2


if __name__ == '__main__':
    pass
