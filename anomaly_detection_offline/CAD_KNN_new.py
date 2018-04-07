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
    :return alp: float
    """
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)
    distance, index = neigh.kneighbors(x_test)
    # print(distance)
    dis = distance.sum(axis=1)
    return dis


def cad_knn_new(x_test, y_test, x_train, y_train, n_nei=3):
    """
    :param x_test: array(1, )
    :param y_test: array
    :param x_train: array
    :param y_train: array
    :param k: int
    :return: array( ,1)
    """
    result = []
    for y_label in y_test:
        nei_sample = []

        for index, label in enumerate(y_train):
            if label == y_label:
                nei_sample.append(x_train[index, :])

        test_alph = k_neighourdis(x_test, nei_sample, k=n_nei)
        test_value = 1
        for train_sample in nei_sample:
            train_alph_sample = k_neighourdis([train_sample], nei_sample, k=n_nei)
            if train_alph_sample >= test_alph:
                test_value += 1
    # count p_value
        p_value = [test_value / (np.shape(nei_sample)[0]), y_label]
        result.append(p_value)
    return result


if __name__ == '__main__':
    a = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0.5, 1]])

    a_y = [1, 1, 1]

    # b = np.array([[1, 1, 1],
    #               [0.5, 1, 1],
    #               [0.5, 0.5, 1],
    #               [1, 0, 0]])
    b = np.array([[0, 0.5, 0.5]])
    b_test = [1]

    result = cad_knn_new(b, b_test, a, a_y, n_nei=1)
    print(result)