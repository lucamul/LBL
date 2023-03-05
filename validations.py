from gc import get_debug
import numpy as np
from helpers import *
import implementations
import data_processing
import predictions
import evaluations

# This function given a seed, N and k_fold returns an array of k_fold indices which belong to the train set (taken from ML labs)
def _build_k_indices(N, k_fold, seed):
    """build k indices for k-fold.

    Args:
        N:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = N
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# Performs one step of cross validation
def _cross_validation_step(
    y, x, k_indices, k, lambda_, max_iters, gamma, loss_bias=1.0
):
    # get dimensions
    _, D = get_dimensions(x)
    # set initial w to 0
    initial_w = np.zeros(D)
    # get the train set by removing the kth index of k_indices from x and y
    train_set = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0)

    # get w by performing regularized logistic regression with the given parameters
    w, _ = implementations.reg_logistic_regression_sgd(
        y_train, train_set, lambda_, initial_w, max_iters, gamma, loss_bias
    )

    # get the test set as the data excluded from the train set
    test_set = x[k_indices[k], :]
    y_test = y[k_indices[k]]
    # calculate the prediction of the test set
    y_hat_test = predictions.logistic_regression(test_set, w)
    # evaluate performance over the test set
    return evaluations.measure_performance(y_test, y_hat_test)


# Performs cross validation on a given parameter
def _cross_validation(parameters, parameter_name, k_fold, seed, N, validation_func):
    # build the k_indices vector for a given k_fold
    k_indices = _build_k_indices(N, k_fold, seed)
    best_performance = []
    optimal_param = -1
    # loop thrpugh all choices for a hyper-parameter
    for parameter in parameters:
        # initial value of performance is 0
        avg_performance = np.array([0, 0, 0, 0, 0])
        performances = np.zeros((k_fold, len(avg_performance)))
        # for each k in k_fold
        for k in range(k_fold):
            # get performance by doing validation
            performance = validation_func(parameter, k_indices, k)
            # update average performance and the array of performances
            avg_performance = avg_performance + performance
            performances[k, :] = performance
        # compute average of performances by dividing by k_fold
        avg_performance = avg_performance / k_fold
        # print the results with this parameter
        print(
            "--> running cross-validation for {} = {}, avg loss {}, with performances {}".format(
                parameter_name, parameter, avg_performance, performances
            )
        )
        # if the obtained performance's F1 score is better than best then update
        if best_performance == [] or avg_performance[-1] > best_performance[-1]:
            best_performance = avg_performance
            optimal_param = parameter
    # return the optimal parameter choice in terms of F1 score
    return optimal_param, best_performance


# Function used to do one cross vaidation step for degree
def _validation_func_over_degree(
    y, x, lambda_, degree, max_iters, gamma, k_indices, k, is_feature_normalization
):
    # clean data
    x = data_processing.clean(x, degree, is_feature_normalization)
    # do step for degree
    return _cross_validation_step(y, x, k_indices, k, lambda_, max_iters, gamma)


# Do cross validation over degree
def cross_validation_over_degree(
    y, x, lambda_, degrees, max_iters, gamma, k_fold, seed, is_feature_normalization
):
    N = len(y)
    validation_func = lambda degree, k_indices, k: _validation_func_over_degree(
        y, x, lambda_, degree, max_iters, gamma, k_indices, k, is_feature_normalization
    )
    return _cross_validation(degrees, "degree", k_fold, seed, N, validation_func)


# Do cross validation over lambda
def cross_validation_over_lambda(y, x, lambdas, degree, max_iters, gamma, k_fold, seed):
    N = len(y)
    validation_func = lambda lambda_, k_indices, k: _cross_validation_step(
        y, x, k_indices, k, lambda_, max_iters, gamma
    )
    return _cross_validation(lambdas, "lambda", k_fold, seed, N, validation_func)


# Do cross validation over gamma
def cross_validation_over_gammas(
    y, x, lambda_, degree, max_iters, gammas, k_fold, seed
):
    N = len(y)
    validation_func = lambda gamma, k_indices, k: _cross_validation_step(
        y, x, k_indices, k, lambda_, max_iters, gamma
    )
    return _cross_validation(gammas, "gamma", k_fold, seed, N, validation_func)


# Do cross validation over loss_bias
def cross_validation_over_loss_bias(
    y, x, lambda_, degree, max_iters, gamma, k_fold, seed, loss_biases
):
    N = len(y)
    validation_func = lambda bias, k_indices, k: _cross_validation_step(
        y, x, k_indices, k, lambda_, max_iters, gamma, bias
    )
    return _cross_validation(loss_biases, "loss bias", k_fold, seed, N, validation_func)
