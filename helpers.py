# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

# For tx of size NxD return N,D
def get_dimensions(tx):
    N = np.shape(tx)[0]
    D = np.shape(tx)[1]
    return N, D


# Function used to create the batches for SGD of size batch_size. (This function was taken from the ML labs)
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset. If batch is equal to size of dataset.
    The functions creates num_batches time a batch of the complete dataset.

    In the case where batch size is smaller than the dataset, the function takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if batch_size == data_size:
        for _ in range(num_batches):
            yield y, tx
    else:
        batch_size = int(batch_size)
        if shuffle:
            np.random.seed(496)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[
                    start_index:end_index
                ]
