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
    # V_mat contains all of the small w values for each hidden unite, layer 1
    # w_vec contains all of the small w values for y_hat, layer 2

    # m x n_hidden_units matrix
    V_mat = stats.zscore(np.random.random_sample(X_subtrain.shape[1], n_hidden_units))
    # 
    w_vec = stats.zscore(np.random.random_sample(X_subtrain.shape[1]))
    h_vec = stats.zscore(np.random.random_sample(n_hidden_units))
    
    for i in range(is_subtrain.shape[0]):
        if(is_subtrain[i] == True):
            X_train.append(np.copy(X_mat[i]))
    for epoch in range(max_epochs):
        for row in range(X_subtrain.shape[0]):
            w_vec = Forward_Propagation(V_mat)
            w_vec = Back_Propagation(X_subtrain[row], h_vec, V_mat)
    return V_mat, w_vec

def Back_Propagation(data_point, h_vec, V_mat):
    h_vec = grad_a
    for l in range(2):
        if(l == 1):
            grad_a[l] = 
        else:
            grad_h[l] = grad_a[l] * h[l] * (1 - h[l])
            grad_a[l] = np.matmul(grad_a[l], h[l - 1].T)
    return grad_w
            

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
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    return temp_ar

if len(sys.argv) < 4:
    help_str = """Execution example: python3 main.py <No.of folds k> <No. of Nearest neighbors> <seed>
Folds must be a float
NN must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

num_folds = int(sys.argv[1])
max_neighbors = int(sys.argv[2])
seed = int(sys.argv[3])
np.random.seed(seed)
temp_ar = Parse("spam.data")

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)
