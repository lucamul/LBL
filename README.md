### Introduction
The purpose of the experiment is to use machine learning tecniques on data from the CERN accelerator to distinguish between the Higgs boson or background.
Under the folder /code, the following scripts are included:<br />
&emsp;-data_processing.py: this script deals with the processing of data, for example loading the data set and normalizing it;<br />
&emsp;-evaluations.py: the script contains functions that evaluate the obtained results. Losses, accuracy and other evaluations are calculated here;<br />
&emsp;-gradients.py: the script implements simple functions used by the implementations.py script (e.g. stochastic gradient descent, logistic regression etc.);<br />
&emsp;-helpers.py: helper functions are cointained here, for example the one that creates mini-batches;<br />
&emsp;-predictions.py: some other helper functions are in this script, more related to the prediction part (e.g. the sigmoid function);<br />
&emsp;-validations.py: it implements cross-validation functions to find important hyper-parameters (lambda, degree etc.);<br />

The implementations.py file does most of the heavy lifting, it containes the six functions that are asked to be implemented: (GD, SGD, least_square, ridge_regression logistic_regression, reg_logistic_regression).

The run.py file is used to run experiments on this code base:<br />
&emsp;-) ```python3 run.py -h``` provides the user with an overview of the options;<br />
&emsp;-) ```python3 run.py -v``` runs the codebase with cross validation over gamma, lambda, degree and loss_bias;<br />
&emsp;-) ```python3 run.py -n``` implements feature normalization;<br />
&emsp;-) ```python3 run.py --lambda=0.3 --gamma=0.2 --degree=1``` allows to run the code setting lambda, gamma and degree to any value (remark: this is only useful when not running with cross validation). <br />
&emsp;-) ```python3 run.py -lbd``` these three options allow to run cross-validation over lambda, loss_bias and degree respectively <br />
&emsp;-) ```python3 run.py -r``` replace n.d. data with the mean.

### Reproduction of the report results
Note that all our experiments run with stochastic gradient descent of batch size 1. We used a seed (496) to make the experiments reproducible. However, the result may vary based on the machine used.<br />
&emsp;-) Experiment 1: ```python3 run.py --lambda=0 --degree=1```<br />
&emsp;-) Experiment 2: ```python3 run.py --lambda=0 --degree=1 -r``` <br />
&emsp;-) Experiment 3: ```python3 run.py -nr --degree=1 --lambda=0``` <br />
&emsp;-) Experiment 4: ```python3 run.py -nrl --degree=1``` <br />
&emsp;-) Experiment 5: ```python3 run.py -nrld``` <br />
&emsp;-) Experiment 6: ```python3 run.py -vrn``` <br />
