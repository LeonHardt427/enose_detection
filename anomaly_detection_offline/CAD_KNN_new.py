# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:22
# @Author  : LeonHardt
# @File    : CAD_KNN_new.py

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


def k_neighourdis(train, x_test, k=3):
    """
    :param x_test: array (1,)
    :param train: array(:,:)
    :param k: int
    :return alp: float
    """
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)
    distance, index = neigh.kneighbors(x_test)
    # print(distance)
    dis = distance.sum(axis=1)[0]
    return dis


def cad_knn_new(x_train, y_train, x_test, poss_labels, k_nei=1):
    """
    计算p_value（Alpha包含其中）
    :param x_test: array(:, :)
    :param poss_labels: []
    :param x_train: array(:, :)
    :param y_train: array(:,:)
    :param k_nei: int
    :return: array((x_test.shape[0], len(poss_labels))
    """
    p_value = np.zeros((x_test.shape[0], len(poss_labels)))
    for num, x_sample in enumerate(x_test):    # count sample one by one

        test_pvalue = np.zeros(len(poss_labels))

        for y_index, y_label in enumerate(poss_labels):   # extract same label train samples
            same_samples = []
            flag = 0            # same_samples the first line
            for index, label in enumerate(y_train):
                if label == y_label:
                    if flag == 0:
                        same_samples = x_train[index, :]
                        flag = 1
                    else:
                        same_samples = np.vstack((same_samples, x_train[index, :]))
                        flag = 2

            sample_whole = same_samples             # prepare for count train alpha
            if flag == 1 :
                test_alph = k_neighourdis(np.array([same_samples]), np.array([x_sample]), k=k_nei)    # a_test
            elif flag == 2:
                test_alph = k_neighourdis(np.array(same_samples), np.array([x_sample]), k=k_nei)  # a_test
            train_alph = []
            sample_whole = np.vstack((sample_whole, x_sample))

            if flag == 1:
                same_samples = np.array([same_samples])

            for same_sample in same_samples:            # count all a_train
                train_alph_temp = k_neighourdis(sample_whole, np.array([same_sample]), k=k_nei+1)
                train_alph.append(train_alph_temp)

            test_pvalue_sample = 1
            for n, alpha in enumerate(train_alph):
                if alpha >= test_alph:
                    test_pvalue_sample += 1
            test_pvalue_sample = (test_pvalue_sample / sample_whole.shape[0])

            test_pvalue[y_index] = test_pvalue_sample

        p_value[num, :] = test_pvalue

    return p_value


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

    result = cad_knn_new(x_train=a, y_train=a_y, x_test=b, poss_labels=b_test, k_nei=1)
    # result = k_neighourdis(a, b, k=1)
    print(result)