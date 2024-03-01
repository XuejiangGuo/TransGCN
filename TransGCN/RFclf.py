# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : RFclf.py
# @Time     : 5/25/2023 10:34 PM
# @Emal     : wangbing587@163.com
# @Desc     :


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def RFclf(x, y, x_test, val_size=0.2, seed=0):
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y,
                                                      test_size=val_size, random_state=seed)
    clf = RandomForestClassifier(n_jobs=-1, random_state=seed, class_weight='balanced')
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test), round(clf.score(x_val, y_val), 4)


if __name__ == '__main__':
    pass

