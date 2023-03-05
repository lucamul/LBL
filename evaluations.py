import numpy as np

# Calculates the MSE loss
def mse_loss(y, y_hat):
    # get N as a dimension of tx
    N = len(y)
    # calculate y - (tx)(w)
    r = np.subtract(y, y_hat)
    # return (1/(2N)) times the squared norm of r
    return (0.5 / N) * np.dot(np.transpose(r), r)[0, 0]


# Calculates the logistic loss
def logistic_loss(y, y_hat):
    eps = 1e-15
    # get the size og y
    N = y.shape[0]
    # perform the sum and multiply by 1/N
    return (
        1.0 / N * np.sum(-y * np.log(y_hat + eps) - (1 - y) * np.log(1 - y_hat + eps))
    )


# Calculates accuracy of y_hat
def accuracy(y, y_hat):
    # take the y_hat and get the classification (1,0)
    predictions = np.floor(y_hat + 0.5)
    # return num of correct predictions / total predictions
    return np.sum(y == predictions) / len(y)


# Calculates precision of y_hat
def precision(y, y_hat):
    # take the y_hat and get the classification (1,0)
    predictions = np.floor(y_hat + 0.5)
    # return number of true positives divided by number of true positives + number of false positives
    return np.sum(y * predictions) / np.sum(predictions)


# Calculates recall of y_hat
def recall(y, y_hat):
    # take the y_hat and get the classification (1,0)
    predictions = np.floor(y_hat + 0.5)
    # return number of true positive divided by number of true positives + number of false negatives
    return np.sum(y * predictions) / np.sum(y)


# Calculates the f1 score
def f1_score(y, y_hat):
    # get precision
    p = precision(y, y_hat)
    # get recall
    r = recall(y, y_hat)
    # F1 = 2(pr)/(p+r)
    return 2 * (p * r) / (p + r)


# Returns a measurement of pergormance as loss, accuracy, precision, recall and f1 score
def measure_performance(y, y_hat):
    acc = accuracy(y, y_hat)
    prec = precision(y, y_hat)
    rec = recall(y, y_hat)
    f1 = f1_score(y, y_hat)
    loss = logistic_loss(y, y_hat)
    return np.array([loss, acc, prec, rec, f1])
