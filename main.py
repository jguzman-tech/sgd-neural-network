# This Python file uses the following encoding: iso-8859-1

import os, sys
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path
import copy
import warnings
import statistics
warnings.simplefilter('error') # treat warnings as errors

from matplotlib.pyplot import figure
figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)
from gradient import Gradient, GradientDescent, MeanLogisticLoss

def NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain):
    X_subtrain = np.array([[]])
    n_col = X_mat.shape[1]
    # V_mat contains all of the small w values for each hidden unite, layer 1
    # w_vec contains all of the small w values for y_hat, layer 2
    V_mat = stats.zscore(np.random.random_sample(n_col, n_hidden_units)) / 10
    w_vec = stats.zscore(np.random.random_sample(n_col)) / 10
    # create a weight  list and put two weight in this list
    weight_list = list();
    weight_list.append(V_mat);
    weight_list.append(w_vec);
    for i in range(is_subtrain.shape[0]):
        if(is_subtrain[i] == True):
            X_train.append(np.copy(X_mat[i]))
    for epoch in range(max_epochs):
        for row in range(X_subtrain.shape[0]):
            # forward propagation
            observation = X_subtrain[row]
            h_list = ForwardPropagation(observation, weight_list)
    
    return h_list
    #return V_mat, w_vec

# forward propagation function
def ForwardPropagation(X, weight_list)
    h_list = list()
    h_list.append(X)
    for i in range(len(weight_list))
        a_vec = np.matmul(X, weight_list[i])
        if(i == len(weight_list))
            h_list.append(a_vec)
        else
            h_vec = 1/(1+exp(-a_vec))
            h_list.append(h_vec)

    return h_list

def Parse(fname):
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])`
    return temp_ar

if len(sys.argv) < 4:
    help_str = """Execution example: python3 main.py <No.of folds k> <No. of Nearest neighbors> <seed>
Folds must be a float
NN must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

step_size = int(sys.argv[1])
n_hidden_units = int(sys.argv[2])
seed = int(sys.argv[3])
np.random.seed(seed)
temp_ar = Parse("spam.data")

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)
num_row = X.shape[0]

#Next create a variable is.train (logical vector with size equal to the number of observations in the whole data set). 
is_train =  np.random.shuffle(np.repeat(("TRUE", "FALSE"), [num_row * 0.8, num_row * 0.2], axis = 0))

#Next create a variable is.subtrain (logical vector with size equal to the number of observations in the train set). 
num_train = num_row * 0.8
is_subtrain = np.random.shuffle(np.repeat(("TRUE", "FALSE"), [num_train * 0.6, num_train * 0.4], axis = 0))

# get train data






