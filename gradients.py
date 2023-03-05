import numpy as np
from helpers import batch_iter, get_dimensions
import predictions

# Computes logistic gradient
def logistic_regression(y, y_hat, tx):
    # return (y_hat - y)*tx which equates to the logistic gradient
    N = y.shape[0]
    return (1 / N) * np.dot(np.transpose(tx), (y_hat - y))


# Computes gradient for regularized logistic regression
def reg_logistic_regression(y, y_hat, tx, lambda_, w):
    # return the gradient of normal logistic regression + 2lambda * 2
    return logistic_regression(y, y_hat, tx) + 2 * lambda_ * w


# Compute MSE gradient
def mse(y, y_hat, tx):
    # get the value of N
    N, _ = get_dimensions(tx)
    # transpose tx
    tx_t = np.transpose(tx)
    # find the error e
    e = y - y_hat
    # return -(1/N)*(tx_t)(e)
    return -(1 / N) * np.dot(tx_t, e)
