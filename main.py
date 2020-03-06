# This Python file uses the following encoding: iso-8859-1

import pdb
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

def NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain):
    train_indices = np.where(is_subtrain == True)[0]
    validation_indices = np.where(is_subtrain == False)[0]
    X_subtrain = X_mat[train_indices]
    y_subtrain = y_vec[train_indices]
    X_validation = X_mat[validation_indices]
    y_validation = y_vec[validation_indices]

    n_col = X_mat.shape[1]
    # V_mat contains all of the small w values for each hidden unite, layer 1
    # w_vec contains all of the small w values for y_hat, layer 2
    V_mat = stats.zscore(np.random.random_sample((n_col, n_hidden_units)))
    w_vec = stats.zscore(np.random.random_sample(n_hidden_units))[np.newaxis].T

    y_tilde_subtrain = np.copy(y_subtrain)
    y_tilde_subtrain[y_tilde_subtrain == 0] = -1

    y_tilde_validation = np.copy(y_validation)
    y_tilde_validation[y_tilde_validation == 0] = -1

    loss_values = {'subtrain': list(), 'validation': list()}

    weight_list = list()
    weight_list.append(None)
    weight_list.append(V_mat)
    weight_list.append(w_vec)
    
    for epoch in range(max_epochs):
        for row in range(X_subtrain.shape[0]):
            observation = X_subtrain[row]
            h_list = ForwardPropagation(observation, weight_list)
            grad_w_list = BackPropagation(h_list, weight_list, y_tilde_subtrain[row])
            # update weight
            for i in range(1, 3):
                weight_list[i] = weight_list[i] - step_size * grad_w_list[i]

        loss_values['subtrain'].append(MeanLogisticLoss(weight_list[1], weight_list[2], X_subtrain, y_tilde_subtrain))
        loss_values['validation'].append(MeanLogisticLoss(weight_list[1], weight_list[2], X_validation, y_tilde_validation))
    return loss_values, V_mat, w_vec

def MeanLogisticLoss(theta1, theta2, X, y_tilde):
    my_sum = 0.0
    n = X.shape[0]
    for row in range(n):
        my_sum += np.log(1 + np.exp(-1 * y_tilde[row, 0] * np.matmul(np.matmul(X[row][np.newaxis], theta1), theta2)[0, 0]))
    return my_sum / n

# forward propagation function
def ForwardPropagation(X, weight_list):
    h_list = [X[np.newaxis].T, None, None]
    for i in range(1, 3):
        if(i == 2): # last layer gets identity
            h_list[i] = np.matmul(weight_list[i].T, h_list[i - 1])
        else:
            h_list[i] = 1 / (1 + np.exp(np.matmul(weight_list[i].T, h_list[i - 1])))
    return h_list

# back progration function
def BackPropagation(h_list, w_list, y_tilde):
    grad_w_list = [None, None, None]
    for i in range(2, 0, -1):
        if(i == 2):
            grad_a = -1 * y_tilde / (1 + np.exp(np.matmul(y_tilde, h_list[i])))
            grad_a = grad_a[np.newaxis]
        else:
            grad_h = np.matmul(w_list[i + 1], grad_a)
            grad_a = grad_h * h_list[i] * (1 - h_list[i])
        grad_w_list[i] = np.matmul(grad_a, h_list[i - 1].T).T

    return grad_w_list

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
    np.random.shuffle(temp_ar)
    return temp_ar

if len(sys.argv) < 4:
    help_str = """Execution example: python3 main.py <No.of folds k> <No. of Nearest neighbors> <seed>
Folds must be a float
NN must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

max_epochs = int(sys.argv[1])
step_size = float(sys.argv[2])
n_hidden_units = int(sys.argv[3])
seed = int(sys.argv[4])
np.random.seed(seed)
temp_ar = Parse("spam.data")

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)
num_row = X.shape[0]

#Next create a variable is.train (logical vector with size equal to the number of observations in the whole data set). 
is_train = np.random.randint(0, 5, X.shape[0]) # 80% train, 20% test (from whole dataset)
is_train = (is_train < 4) # convert to boolean

#Next create a variable is.subtrain (logical vector with size equal to the number of observations in the train set).
is_subtrain = np.random.randint(0, 5, is_train[is_train == True].shape[0]) # 60% subtrain, 40% validation (from training set)
is_subtrain = (is_subtrain < 3) # convert to boolean

# get training set
X_mat = X[np.where(is_train)[0]]
y_vec = y[np.where(is_train)[0]]

# NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain)
loss_values, V_mat, w_vec = NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain)
mll_subtrain = loss_values['subtrain']
mll_validation = loss_values['validation']

plt.plot(mll_subtrain, label='subtrain', color='blue')
plt.plot(mll_validation, label='validation', color='red')
optimal_epoch = mll_validation.index(min(mll_validation))
plt.scatter(optimal_epoch, min(mll_validation), marker='o', edgecolors='r', s=160, facecolor='none', linewidth=3, label='minimum')
plt.legend()
plt.tight_layout()
plt.xlabel('epoch #')
plt.ylabel('Mean Logistic Loss')
info = f"epochs_{max_epochs}_step_{step_size}_units_{n_hidden_units}_seed_{seed}"
fname = f"{info}_logistic_loss.png"
plt.savefig(fname)
