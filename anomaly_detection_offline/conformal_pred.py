# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 16:19
# @Author  : LeonHardt
# @File    : conformal_pred.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
path = os.getcwd()

X = np.loadtxt('dendrobium_x_sample.csv', delimiter=',')
y = np.loadtxt('dendrobium_y_label.csv', delimiter=',', dtype='int8')
x_anomaly = np.loadtxt('x_sample_anomaly.csv', delimiter=',')

sc = StandardScaler()
X = sc.fit_transform(X)

correct = []
anomaly = []

s_fold = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in s_fold.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    lda = LinearDiscriminantAnalysis(n_components=9)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    x_anomaly_lda = lda.transform(x_anomaly)

    x_train, x_cal, y_train, y_cal = train_test_split(X_train_lda,
                                                      y_train, test_size=0.3, shuffle=False, random_state=1)

    model = KNeighborsClassifier(n_neighbors=5)
    # -----------------------------------------------------------------------------
    # Train and calibrate
    # -----------------------------------------------------------------------------

    icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model)))
    icp.fit(x_train, y_train)
    icp.calibrate(x_cal, y_cal)

    # -----------------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------------
    SIG = 0.2
    prediction = icp.predict(X_test_lda, significance=SIG)
    result = np.sum(prediction, axis=1)
    zero_sum_correct = (48 - result.sum(axis=0))/48
    correct.append(zero_sum_correct)
    print("the correct prediction")
    print(result)

    prediction_anomaly = icp.predict(x_anomaly_lda, significance=SIG)
    result_anomaly = np.sum(prediction_anomaly, axis=1)
    zero_sum_anomaly = (50 - result_anomaly.sum(axis=0)) / 50
    anomaly.append(zero_sum_anomaly)
    print("the anomaly prediction")
    print(result_anomaly)

print("correct_summary")
print(correct)
print(np.mean(correct))
print("anomaly_summary")
print(anomaly)
print(np.mean(anomaly))
