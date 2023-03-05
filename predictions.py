# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

# Calculates sigmoid(x) for positive x
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculates sigmoid(x) for negative x
def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


# https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
def _sigmoid(x):
    # get indices for which x is positive and negative
    positive = x >= 0
    negative = ~positive

    result = np.empty_like(x, dtype=float)
    # set the result where x is positive to positive sigmoid and negative to negative sigmoid
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


# Calculates sigmoid((tx)(w))
def logistic_regression(tx, weights):
    y_hat = _sigmoid(tx.dot(weights))
    return y_hat


# Calculates predictions for linear regression
def linear_regression(tx, weights):
    return np.dot(tx, weights)


# Maps the prediciton to 1 and -1, then returns a column array of id-predictions
def _to_predictions(ids, y_hat):
    y_hat[y_hat < 0.5] = -1
    y_hat[y_hat >= 0.5] = 1
    return np.transpose(np.array([ids, y_hat])).astype(int)


# Given w and tx computes the predictions by first getting y_hat and then mapping the result of the sigmoid to the prediction -1, 1
def classify_logistic_regression(ids, tx, weights):
    y_hat = logistic_regression(tx, weights)
    return _to_predictions(ids, y_hat)
