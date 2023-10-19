# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : model.py
# @Time     : 5/12/2023 11:08 PM
# @Emal     : wangbing587@163.com
# @Desc     :


import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *
from loss_function import loss_func
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
from FDR import FDR
from copy import deepcopy

class TransGCN:
    def __init__(self, epochs=5000, nepoch=500, hidden_dim1=256, hidden_dim2=128,
                 drop_rate=0.3, lr=0.01, alpha=2, patience=500, seed=0):
        self.epochs = epochs
        self.nepoch = nepoch
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.drop_rate = drop_rate
        self.lr = lr
        self.alpha = alpha
        self.patience = patience
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, DP):
        print('+++++++++++++Train to Predict Protein Subcellar Localization (PSL)+++++++++++++')
        self.trainPSL(DP.feature, DP.edge,
            DP.y1synsub_label, DP.y1real_rfprob,
            DP.train_mask, DP.val_mask, DP.real_mask)
        print('++++++++++++++++Train to Predict Protein Translocation (PTF)++++++++++++++++')
        self.trainPTF(DP.feature, DP.edge,
            DP.y2synsub_label, DP.y2real_rfprob,
            DP.train_mask, DP.val_mask, DP.real_mask)

        result = deepcopy(DP.df_real)
        markers_SYNprob, markers_SYN = self.realPSL_best.max(1)
        result['markers_SYNprob'] = markers_SYNprob
        result.loc[result['markers'] != 'unknown', 'markers_SYNprob'] = 1.0
        result['markers_SYN'] = markers_SYN
        result['markers_SYN'] = result['markers_SYN'].map(DP.markersSYN_revmap)
        result['markers_T1'] = result['markers_SYN'].apply(lambda x: x[0])
        result['markers_T2'] = result['markers_SYN'].apply(lambda x: x[1])
        result.loc[result['markers'] != 'unknown',
                   'markers_T1'] = result.loc[result['markers'] != 'unknown', 'markers']
        result.loc[result['markers'] != 'unknown',
                   'markers_T2'] = result.loc[result['markers'] != 'unknown', 'markers']
        result['markers_SYN'] = result['markers_T1'].map(str) + 'To' + result['markers_T2'].map(str)

        result['PTFprob'] = self.realPTF_best[:, 1]
        self.result = FDR(result, score_column='PTFprob', label_column='markers')


    def trainPSL(self, feature, edge, y_label, yreal_rfprob,
                 train_mask, val_mask, real_mask):
        ytrain_label, yval_label = y_label[train_mask], y_label[val_mask]
        torch.manual_seed(self.seed)
        model = ssGCN(input_dim=feature.shape[1],
                      hidden_dim1=self.hidden_dim1,
                      hidden_dim2=self.hidden_dim2,
                      output_dim=y_label.unique().shape[0], drop_rate=self.drop_rate).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.valPSL_loss_best = np.Inf
        self.valPSL_losses = []
        self.valPSL_f1scores = []
        counter = 0

        for epoch in range(1, self.epochs + 1):
            output = model(feature.to(self.device), edge.to(self.device)).cpu()
            ytrain_pred, yval_pred, yreal_pred = output[train_mask], output[val_mask], output[real_mask]
            loss = loss_func(y_label_pred=ytrain_pred, y_unlabel_pred=yreal_pred,
                             y_label=ytrain_label, y_unlabel_rfprob=yreal_rfprob,
                             alpha=self.alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = round(loss_func(y_label_pred=yval_pred, y_unlabel_pred=yreal_pred,
                                       y_label=yval_label, y_unlabel_rfprob=yreal_rfprob,
                                       alpha=self.alpha).item(), 4)
            self.valPSL_losses.append(val_loss)
            val_f1score = round(f1_score(yval_label.numpy(),
                                         yval_pred.argmax(dim=1).detach().numpy(), average='macro'), 4)
            self.valPSL_f1scores.append(val_f1score)

            if val_loss < self.valPSL_loss_best:
                counter = 0
                self.epochPSL_best = epoch
                self.valPSL_loss_best = val_loss
                self.valPSL_f1score_best = val_f1score
                self.realPSL_best = torch.exp(output.detach())[real_mask]
            else:
                counter += 1

            if epoch % self.nepoch == 0 or epoch == self.epochs:
                print(f'[{epoch}|{self.epochs}] val_loss: {val_loss} | val_f1score: {val_f1score}')
            if counter >= self.patience:
                print('EarlyStopping counter: {}'.format(counter))
                break

        print(f'BestResult [{self.epochPSL_best}|{self.epochs}] val_loss: {self.valPSL_loss_best} | val_f1score: {self.valPSL_f1score_best}')

    def trainPTF(self, feature, edge, y_label, yreal_rfprob,
                 train_mask, val_mask, real_mask):
        ytrain_label, yval_label = y_label[train_mask], y_label[val_mask]
        torch.manual_seed(self.seed)
        model = ssGCN(input_dim=feature.shape[1],
                      hidden_dim1=self.hidden_dim1,
                      hidden_dim2=self.hidden_dim2,
                      output_dim=y_label.unique().shape[0], drop_rate=self.drop_rate).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.valPTF_loss_best = np.Inf
        self.valPTF_losses = []
        self.valPTF_aucs = []
        counter = 0

        for epoch in range(1, self.epochs + 1):
            output = model(feature.to(self.device), edge.to(self.device)).cpu()
            ytrain_pred, yval_pred, yreal_pred = output[train_mask], output[val_mask], output[real_mask]
            loss = loss_func(y_label_pred=ytrain_pred, y_unlabel_pred=yreal_pred,
                             y_label=ytrain_label, y_unlabel_rfprob=yreal_rfprob,
                             alpha=self.alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = round(loss_func(y_label_pred=yval_pred, y_unlabel_pred=yreal_pred,
                                       y_label=yval_label, y_unlabel_rfprob=yreal_rfprob,
                                       alpha=self.alpha).item(), 4)
            self.valPTF_losses.append(val_loss)
            val_auc = round(roc_auc_score(yval_label.numpy(), yval_pred.detach().numpy()[:, 1]), 4)
            self.valPTF_aucs.append(val_auc)

            if val_loss < self.valPTF_loss_best:
                counter = 0
                self.epochPTF_best = epoch
                self.valPTF_loss_best = val_loss
                self.valPTF_auc_best = val_auc
                self.realPTF_best = torch.exp(output.detach())[real_mask]
            else:
                counter += 1

            if epoch % self.nepoch == 0 or epoch == self.epochs:
                print(f'[{epoch}|{self.epochs}] val_loss: {val_loss} | val_auc: {val_auc}')
            if counter >= self.patience:
                print('EarlyStopping counter: {}'.format(counter))
                break
        print(f'BestResult [{self.epochPTF_best}|{self.epochs}] val_loss: {self.valPTF_loss_best} | val_auc: {self.valPTF_auc_best}')

