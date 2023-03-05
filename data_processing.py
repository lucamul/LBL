# -*- coding: utf-8 -*-
import numpy as np
from helpers import *
import csv


# Loads train data from dataset
def load_train_data(path_dataset):
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, dtype=str)
    return (
        data[:, 0].astype(np.int),
        _to_label(data[:, 1]),
        data[:, 2:].astype(np.float),
    )


# Loads test data from dataset
def load_test_data(path_dataset):
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, dtype=str)
    return data[:, 0].astype(np.int), data[:, 2:].astype(np.float)


# transform the label of the data such that s stands for 1 (found) 0 otherwise
def _to_label(raw_lables):
    return np.array([1 if label == "s" else 0 for label in raw_lables])


# Function used to write the result to a file as "ID prediction"
def write_prediction_result(path, result):
    # open file
    f = open(path, "w")
    # get CSV writer
    writer = csv.writer(f)
    # write header
    writer.writerow(["Id", "Prediction"])
    for row in result:
        # write all predictions
        writer.writerow(row)


# Does feature normalization
def feature_normalization(tx):
    # get dimension D
    _, D = get_dimensions(tx)
    # for each feature
    for i in range(D):
        # set each tx[:,i] to itself minus the mean divided standard deviation
        tx[:, i] = (tx[:, i] - np.mean(tx[:, i])) / np.std(tx[:, i])
    return tx


# Replaces an n.d. value in the data set with the mean
def replace_empty_with_mean(tx):
    tx[tx == -999] = np.nan
    return np.where(np.isnan(tx), np.ma.array(tx, mask=np.isnan(tx)).mean(axis=0), tx)


# Appends a column of 1s to tx
def feature_augmentation(tx):
    N, _ = get_dimensions(tx)
    return np.append(tx, np.ones((N, 1)), axis=1)


# Given an element x[i,j] returns 1, x[i,j], x[i,j]^2 etc.
def build_poly_one_val(xn, degree, phin):
    # if degree == 1 then we don't do any feature expansion
    if degree == 1:
        return xn
    for i in range(degree + 1):
        phin.append(pow(xn, i))
    return phin


# Performs feature expansion
def build_poly(tx, degree):
    # if degree is 1 we don't expand
    if degree == 1:
        return tx
    # get dimensions
    N, D = get_dimensions(tx)
    phi = []
    # for each element i
    for i in range(N):
        phin = []
        # if D = 1 then just expand that value
        if D == 1:
            phin = build_poly_one_val(tx[i], degree)
        else:
            # Expand every feature
            for j in range(D):
                phin = build_poly_one_val(tx[i, j], degree, phin)
        phi.append(phin)
    return np.asarray(phi)


# Cleans the data then augments features
def clean(tx, degree=1, normalize=False, replace_empty=True):
    if replace_empty:
        tx = replace_empty_with_mean(tx)
    if normalize:
        tx = feature_normalization(tx)
    tx = build_poly(tx, degree)
    return feature_augmentation(tx)
