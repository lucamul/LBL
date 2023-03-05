import sys, getopt
import implementations
import validations
import data_processing
import predictions
import helpers
import numpy as np

# default path of training, sanity check and test data
training_data_path = "./data/train.csv"
sanity_ceck_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"

# default values for iterations, CV iteration and whether to write result, run CV, feature normalization or replace n.d. values
max_iters = 250000
cv_max_iters = 200000
run_cross_validation = False
replace_empty = False
write_result = True

is_feature_normalization = False

# default values of gamma, lambda and degree
gamma_ = 0.0008
lambda_ = 0.0002
degree = 2


# search space for hyper-parameters
lambdas = [
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
    0.001,
]
degrees = [1, 2, 3, 4, 5]
gammas = [
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
]
loss_biases = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.918,
    2,
]

# default seed and k_fold
seed = 496
k_fold = 5


def run(argv):
    # get options
    opts, args = getopt.getopt(
        argv, "vnhrlbd", ["trainfile=", "testfile=", "gamma=", "lambda=", "degree="]
    )

    # use global variables
    global run_cross_validation
    global training_data_path
    global is_feature_normalization
    global test_data_path
    global gamma_
    global lambda_
    global degree
    global replace_empty

    # by default we don't run CV on degree, lambda and loss_bias but we do for gamma
    is_gamma_set = False
    run_cv_degree = False
    run_cv_lambda = False
    run_cv_loss_bias = False

    # read options
    for opt, arg in opts:
        if opt == "-h":
            print("--trainfile=/path/to/train/data --testfile=/path/to/testdata")
            print("--gamma=float --lambda=float --degree=int")
            print("-v: to run cross validation on all paramters")
            print("-n: to run feature normalization")
            print("-r: to do the replacement of n.d. with the mean")
            print("-l: run cross validation over lambda")
            print("-b: run cross validation over loss bias")
            print("-d: run cross validation over degree")
            exit(1)
        if opt == "-l":
            run_cv_lambda = True
        if opt == "-b":
            run_cv_loss_bias = True
        if opt == "-d":
            run_cv_degree = True
        if opt == "--trainfile":
            training_data_path = arg
        if opt == "--testfile":
            test_data_path = arg
        elif opt == "-v":
            run_cross_validation = True
        elif opt == "-r":
            replace_empty = True
        elif opt == "--gamma":
            is_gamma_set = True
            gamma_ = float(arg)
        elif opt == "--lambda":
            lambda_ = float(arg)
        elif opt == "--degree":
            degree = int(arg)
        elif opt == "-n":
            is_feature_normalization = True

    # load training data
    train_id, y, raw = data_processing.load_train_data(training_data_path)

    # if told to run CV for degree
    if run_cross_validation or run_cv_degree:
        best_degree, degree_performance = validations.cross_validation_over_degree(
            y,
            raw,
            lambda_,
            degrees,
            cv_max_iters,
            gamma_,
            k_fold,
            seed,
            is_feature_normalization,
        )
        print(
            "===> found optimal degree ({}) with average performance: ({})".format(
                best_degree, degree_performance
            )
        )
    else:
        # if not told set degree to 1
        best_degree = 1
    # get tx by cleaning data with the given degree
    tx = data_processing.clean(
        raw, best_degree, is_feature_normalization, replace_empty
    )
    # run cross validation over gamma unless a gamma is given
    if run_cross_validation or not is_gamma_set:
        best_gamma, gamma_performance = validations.cross_validation_over_gammas(
            y, tx, lambda_, best_degree, cv_max_iters, gammas, k_fold, seed
        )
        print(
            "===> found optimal gamma ({}) with average performance: ({})".format(
                best_gamma, gamma_performance
            )
        )
    else:
        best_gamma = gamma_
    # run CV over lambda if told to
    if run_cross_validation or run_cv_lambda:
        best_lambda, lambda_performance = validations.cross_validation_over_lambda(
            y, tx, lambdas, best_degree, cv_max_iters, best_gamma, k_fold, seed
        )
        print(
            "===> found optimal lambda ({}) with average performance: ({})".format(
                best_lambda, lambda_performance
            )
        )
    else:
        # else set lambda to 0 (no regularization)
        best_lambda = 0
    # run CV over loss bias if told to
    if run_cross_validation or run_cv_loss_bias:
        (
            best_loss_bias,
            loss_bias_performance,
        ) = validations.cross_validation_over_loss_bias(
            y,
            tx,
            best_lambda,
            best_degree,
            cv_max_iters,
            best_gamma,
            k_fold,
            seed,
            loss_biases,
        )
        print(
            "===> found optimal loss bias ({}) with average performance: ({})".format(
                best_loss_bias, loss_bias_performance
            )
        )
    else:
        # else set loss bias to 0
        best_loss_bias = 0
    _, D = helpers.get_dimensions(tx)
    initial_w = np.zeros(D)
    # train with the found hyper parameters
    weights, performance = implementations.reg_logistic_regression_sgd(
        y, tx, best_lambda, initial_w, max_iters, best_gamma, best_loss_bias
    )

    print(
        "===> training performance: ({}) with "
        "(lambda: {}), (gamma: {}), (degree: {}), (loss bias: {}), (fn: {}), (seed: {})".format(
            performance,
            best_lambda,
            best_gamma,
            best_degree,
            best_loss_bias,
            is_feature_normalization,
            seed,
        )
    )

    # sanity check
    train_result = predictions.classify_logistic_regression(train_id, tx, weights)

    sanity_id, _, sanityx = data_processing.load_train_data(sanity_ceck_data_path)
    sanityx = data_processing.clean(
        sanityx, best_degree, is_feature_normalization, replace_empty
    )
    sanity_result = predictions.classify_logistic_regression(
        sanity_id, sanityx, weights
    )
    np.testing.assert_array_equal(train_result, sanity_result)

    # load test data and perform predictions
    test_id, ttx = data_processing.load_test_data(test_data_path)
    ttx = data_processing.clean(
        ttx, best_degree, is_feature_normalization, replace_empty
    )
    result = predictions.classify_logistic_regression(test_id, ttx, weights)

    # write results if told to
    if write_result:
        test_predictions_path = "./data/predictions_g={}_it={}_f1={}.csv".format(
            best_gamma, max_iters, performance[-1]
        )
        data_processing.write_prediction_result(test_predictions_path, result)

    return


if __name__ == "__main__":
    run(sys.argv[1:])
