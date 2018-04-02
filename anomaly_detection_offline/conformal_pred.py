# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 16:19
# @Author  : LeonHardt
# @File    : conformal_pred.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
path = os.getcwd()

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

sc = StandardScaler()
X = sc.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

model = SVC(probability=True)
# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(x_train, y_train, test_size=0.3, shuffle=True)

icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model),
                                 MarginErrFunc()))
icp.fit(x_train, y_train)
icp.calibrate(x_cal, y_cal)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = icp.predict(x_test, significance=0.5)
# print(prediction)
result = np.sum(prediction, axis=1)
print(result)

