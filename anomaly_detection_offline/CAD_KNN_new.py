# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:22
# @Author  : LeonHardt
# @File    : CAD_KNN_new.py

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


def k_neighourdis(x_test, train, k=3):
    """
    :param x_test: numpy
    :param train: numpy
    :param k: int
    :return alp: []
    """
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)
    distance, index = neigh.kneighbors(x_test)
    # print(distance)
    dis = distance.sum(axis=1)
    return dis


def cad_knn_new(x_test, y_test, x_train, y_train, n_nei=3):
    """
    计算p_value（Alpha包含其中）
    :param x_test: array(1, )
    :param y_test: array
    :param x_train: array
    :param y_train: array
    :param k: int
    :return: array( ,1)
    """
    result = []
    p_value = []
    for y_index, y_label in enumerate(y_test):
        nei_sample = []   # same label train samples

        for index, label in enumerate(y_train):
            if label == y_label:
                nei_sample.append(x_train[index, :])

        test_alphs = k_neighourdis(x_test, nei_sample, k=n_nei)    # all test samples alpha
        test_alph_temp = np.ones(test_alphs.shape)

        for train_sample in nei_sample:
            train_alph_sample = k_neighourdis([train_sample], nei_sample, k=n_nei)
            for index, test_alph in enumerate(test_alphs):
                if train_alph_sample >= test_alph:
                    test_alph_temp[index] += 1
    # count p_value for one label
        for num, alpha in enumerate(test_alph_temp):
            if num == 0:
                p_value_temp = [alpha / (np.shape(nei_sample)[0])]
            else:
                p_value_temp.append((alpha / (np.shape(nei_sample)[0])))

    # summary p_value
        if y_index == 0:
            p_value = np.array(p_value_temp).T
        else:
            p_value = np.hstack((p_value, np.array(p_value_temp).T))

    return result


if __name__ == '__main__':
    a = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0.5, 1]])

    a_y = [1, 0, 1]

    b = np.array([[1, 1, 1],
                  [0.5, 1, 1],
                  [0.5, 0.5, 1],
                  [1, 0, 0]])
    # b = np.array([[0, 0.5, 0.5]])
    b_test = [0, 1]

    result = cad_knn_new(b, b_test, a, a_y, n_nei=1)
    print(result)