# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:19
# @Author  : LeonHardt
# @File    : anomaly_offline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
save_path = os.getcwd() + '/ICP_result/'
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

X = np.loadtxt('dendrobium_x_sample.csv', delimiter=',')
y = np.loadtxt('dendrobium_y_label.csv', delimiter=',', dtype='int8')
x_anomaly = np.loadtxt('x_sample_anomaly.csv', delimiter=',')

signs = np.arange(0.05, 1, 0.05)

sc = StandardScaler()
X = sc.fit_transform(X)

correct_summary = []
anomaly_summary = []

s_fold = StratifiedKFold(n_splits=10, shuffle=True)
for index, (train_index, test_index) in enumerate(s_fold.split(X, y)):
    correct = []
    anomaly = []

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    lda = LinearDiscriminantAnalysis(n_components=9)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    x_anomaly_lda = lda.transform(x_anomaly)

    x_train, x_cal, y_train, y_cal = train_test_split(X_train_lda,
                                                      y_train, test_size=0.3, shuffle=False, random_state=1)

    model = KNeighborsClassifier(n_neighbors=6)
    # -----------------------------------------------------------------------------
    # Train and calibrate
    # -----------------------------------------------------------------------------

    icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model)))
    icp.fit(x_train, y_train)
    icp.calibrate(x_cal, y_cal)

    # -----------------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------------
    for sig in signs:
        print("sig=" + str(sig))
        SIG = sig
        prediction = icp.predict(X_test_lda, significance=SIG)
        result = np.sum(prediction, axis=1)
        result = result > 0
        zero_sum_correct = (result.sum(axis=0))/48
        correct.append(zero_sum_correct)
        print("the correct prediction")
        print(result)

        prediction_anomaly = icp.predict(x_anomaly_lda, significance=SIG)
        result_anomaly = np.sum(prediction_anomaly, axis=1)
        result_anomaly = result_anomaly > 0
        zero_sum_anomaly = (result_anomaly.sum(axis=0)) / 50
        anomaly.append(zero_sum_anomaly)
        print("the anomaly prediction")
        print(result_anomaly)

    if index == 0:
        correct_summary = correct
        anomaly_summary = anomaly
    else:
        correct_summary = np.vstack((correct_summary, correct))
        anomaly_summary = np.vstack((anomaly_summary, anomaly))

np.savetxt(save_path+'ICP_prediction.txt', correct_summary, delimiter=',')
np.savetxt(save_path+'ICP_anomaly.txt', anomaly_summary, delimiter=',')

correct_plt = np.mean(correct_summary, axis=0)
anomaly_plt = np.mean(anomaly_summary, axis=0)

fig = plt.figure(figsize=(12, 16))
ax1 = plt.subplot(1, 1, 1)

ax1.plot(signs, correct_plt, linestyle='-', color='blue', label='normal_sample')
ax1.plot(signs, anomaly_plt, linestyle='-', color='red', label='anomaly_sample')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc='best')
ax1.set_title('ICP_KNN anomaly detection')
ax1.set_xlabel("Significance", fontsize=10, fontweight='bold')
ax1.set_ylabel("Average count", fontsize=10, fontweight='bold')

plt.show()




