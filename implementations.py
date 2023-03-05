import numpy as np
import sys
from math import sqrt
import gradients
from helpers import *
import evaluations
import predictions

# Performs Gradient Descent for max_iters steps:
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of GD to do
    gamma: value of gamma to use for GD

    returns:
    final weights and loss
    """
    N = len(y)
    return _mean_squared_error(y, tx, initial_w, max_iters, gamma, N)


# Performs stochastic gradient descent (SGD) for max_iters iterations
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of SGD to do
    gamma: value of gamma to use for SGD
    batch_size: size of the batch to use for SGD

    returns final weights and loss
    """
    return _mean_squared_error(y, tx, initial_w, max_iters, gamma, batch_size)


# Performs means squared optimization
def _mean_squared_error(y, tx, initial_w, max_iters, gamma, batch_size):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of SGD to do
    gamma: value of gamma to use for SGD
    batch_size: size of the batch to use for SGD

    returns final weights and loss
    """
    # set w to an initial value (normally an array of 0s)
    w = initial_w
    loss = -1
    # for each iteration i
    for minibatch_y, minibatch_tx in batch_iter(
        y, tx, batch_size=batch_size, num_batches=max_iters
    ):
        # G is the stochastic gradient associated to y, tx, wi
        minibatch_y_hat = predictions.linear_regression(minibatch_tx, w)
        G = gradients.mse(minibatch_y, minibatch_y_hat, minibatch_tx)
        # compute w(i+1) using G
        w = w - gamma * G
    # loss is the MSE loss associated to y, yx, wi
    y_hat = predictions.linear_regression(tx, w)
    loss = evaluations.mse_loss(y, y_hat)
    # return w_(max_iters) and its associated MSE loss
    return w, np.array(loss)


# Finds the optimal w for (y,tx) by using the least squares method
def least_squares(y, tx):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data

    returns final weights and loss
    """
    # compute the transpose of tx
    tx_t = np.transpose(tx)
    # find w as the solution for w of the linear system tx(tx_t)w = (tx_t)y
    w = np.linalg.solve(tx_t.dot(tx), tx_t.dot(y))
    # return w and its associated MSE loss
    y_hat = predictions.linear_regression(tx, w)
    return w, np.array(evaluations.mse_loss(y, y_hat))


# Performs ridge regression for a given lambda value
def ridge_regression(y, tx, lamba):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    lambda: value of lambda to use for regularization

    returns:
    weights and loss
    """
    # get the dimensions of tx
    N, D = get_dimensions(tx)
    # transpose tx
    tx_t = np.transpose(tx)
    # obtain A = (tx_t)t + 2*N*lambda*(D*I)
    A = tx_t.dot(tx) + ((2 * N) * (lamba)) * np.eye(D)
    # solve the linear system Aw = (tx_t)y for w
    w = np.linalg.solve(A, tx_t.dot(y))
    # return w and its associated MSE loss
    y_hat = predictions.linear_regression(tx, w)
    return w, np.array(evaluations.mse_loss(y, y_hat))


# logistic regression with gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of GD to do
    gamma: value of gamma to use for GD

    returns:
    weights and loss
    """
    N = len(y)
    gradient_func = lambda y, y_hat, tx, weights: gradients.logistic_regression(
        y, y_hat, tx
    )
    return _logistic_regression(
        y, tx, initial_w, max_iters, gamma, gradient_func, batch_size=N, loss_bias=0.0
    )


# regularized logistic regression with gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    lambda_: value used for regularization
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of GD to do
    gamma: value of gamma to use for GD

    returns:
    weights and loss
    """
    if lambda_ == 0:
        return logistic_regression(y, tx, initial_w, max_iters, gamma)

    N = len(y)
    gradient_func = lambda y, y_hat, tx, weights: gradients.reg_logistic_regression(
        y, y_hat, tx, lambda_, weights
    )
    return _logistic_regression(
        y, tx, initial_w, max_iters, gamma, gradient_func, batch_size=N, loss_bias=0.0
    )


# logistic regression with stochastic gradient descent
def logistic_regression_sgd(
    y, tx, initial_w, max_iters, gamma, loss_bias=0.0, batch_size=1
):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of SGD to do
    gamma: value of gamma to use for SGD
    batch_size: size of the batch to use for SGD
    loss_bias: value of loss bias

    returns:
    weights and loss
    """
    N = len(y)
    gradient_func = lambda y, y_hat, tx, weights: gradients.logistic_regression(
        y, y_hat, tx
    )
    w, _ = _logistic_regression(
        y,
        tx,
        initial_w,
        max_iters,
        gamma,
        gradient_func,
        batch_size=batch_size,
        loss_bias=loss_bias,
    )
    y_hat = predictions.logistic_regression(tx, w)
    performance = evaluations.measure_performance(y, y_hat)
    return w, performance


# regularized logistic regression with stochastic gradient descent
def reg_logistic_regression_sgd(
    y, tx, lambda_, initial_w, max_iters, gamma, loss_bias=0.0, batch_size=1
):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    lambda_: value used for regularization
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of SGD to do
    gamma: value of gamma to use for SGD
    loss_bias: value of loss bias
    batch_size: size of the batch to use for SGD

    returns:
    weights and loss
    """
    if lambda_ == 0:
        return logistic_regression_sgd(
            y,
            tx,
            initial_w,
            max_iters,
            gamma,
            loss_bias=loss_bias,
            batch_size=batch_size,
        )

    N = len(y)
    gradient_func = lambda y, y_hat, tx, weights: gradients.reg_logistic_regression(
        y, y_hat, tx, lambda_, weights
    )
    w, _ = _logistic_regression(
        y,
        tx,
        initial_w,
        max_iters,
        gamma,
        gradient_func,
        batch_size=batch_size,
        loss_bias=loss_bias,
    )
    y_hat = predictions.logistic_regression(tx, w)
    performance = evaluations.measure_performance(y, y_hat)
    return w, performance


def _logistic_regression(
    y, tx, initial_w, max_iters, gamma, gradient_func, batch_size, loss_bias
):
    """
    y (N,): vector of classifications of input data
    tx (N,D): matrix of input data
    initial_w (D,): vector of initial values of w
    max_iters: number of iterations of GD to do
    gamma: value of gamma to use for GD
    gradient_func: function to use for gradient calculation
    batch_size: size of the batch to use for GD
    loss_bias: value of loss bias

    returns:
    weights and loss
    """
    if loss_bias != 0.0:
        assert batch_size == 1, "loss_bias can only be used using SGD with batch size 1"

    # get the dimensions N, D
    N, D = tx.shape
    # set w to an initial value (usually a vector of 0s)
    weights = initial_w
    # do stochastic gradient descent: for each iteration select a subset of size batch_size of y and tx
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        minibatch_y_hat = predictions.logistic_regression(minibatch_tx, weights)
        grad = gradient_func(minibatch_y, minibatch_y_hat, minibatch_tx, weights)
        weights = weights - gamma * grad
        if loss_bias != 0.0:
            weights = weights - loss_bias * minibatch_y * gamma * grad
    y_hat = predictions.logistic_regression(tx, weights)
    loss = evaluations.logistic_loss(y, y_hat)
    # return final w value and the associated logistic loss
    return weights, loss
