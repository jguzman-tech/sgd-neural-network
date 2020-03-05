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
    w_vec = stats.zscore(np.random.random_sample((n_hidden_units, 1)))

    y_tilde = np.copy(y_vec)
    y_tilde[y_tilde == 0] = -1

    # the loss value
    loss_value = list()
    # epoch part
    for epoch in range(max_epochs):
        # list of weight_list for each row
        weight_list_X = list()
        for row in range(X_subtrain.shape[0]):
            # create a weight  list and put two weight in this list
            weight_list = list()
            weight_list.append(V_mat)
            weight_list.append(w_vec)
            # forward propagation
            observation = X_subtrain[row]
            h_list = ForwardPropagation(observation, weight_list)
            # back propagation
            
            grad_w_list = Back_Propagation(h_list, weight_list, y_tilde[row])
            
            # update the theta for each weight
            for i in range(len(weight_list)):
                weight_list[i] = weight_list[i] - step_size * grad_w_list[ 1 - i]
            print(weight_list)
            weight_list_X.append(weight_list)
        
        ## compute the logistic loss for subtraihttps://forum.handsontable.com/t/adding-dictionary-map-array-as-data/4199/3n and validaiton
        # get prediction for subtrain and validaiton
    #return h_list
    return V_mat, w_vec

# logistic loss function
def LogisticLoss(pred, label):
    value = np.log(1 + np.exp(-label * pred))
    return value

# forward propagation function
def ForwardPropagation(X, weight_list):
    h_list = list()
    h_list.append(X)# is it the correct way to append a numpy array into a list? or use np.copy(X)
    h_vec = X[np.newaxis]
    for i in range(len(weight_list)):
        a_vec = np.matmul(h_vec, weight_list[i])
        if(i == len(weight_list)):
            h_list.append(a_vec)# is it the correct way to append a numpy array into a list?
        else:
            h_vec = 1/(1+np.exp(-a_vec))
            h_list.append(h_vec)# is it the correct way to append a numpy array into a list?

    return h_list

# back progration function
def Back_Propagation(h_list, w_list, y_tilde):
    grad_w_list = list()
    for i in range(2, 0, -1):
        if(i == 2):
            grad_a = -1 * y_tilde / (1 + np.exp(y_tilde * h_list[i]))
        else:
            grad_h = np.matmul(grad_a, w_list[i].T)

            grad_a = grad_h * h_list[i] * (1 - h_list[i])
        
        grad_w_list.append(np.matmul(h_list[i - 1][np.newaxis].T, grad_a))

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
is_train = np.random.randint(0, 5, X.shape[0]) # 80% train, 20% test (from whole dataset)
is_train = (is_train < 4) # convert to boolean

#Next create a variable is.subtrain (logical vector with size equal to the number of observations in the train set).
is_subtrain = np.random.randint(0, 5, is_train[is_train == True].shape[0]) # 60% subtrain, 40% validation (from training set)
is_subtrain = (is_subtrain < 3) # convert to boolean

# get training set
X_mat = X[np.where(is_train)[0]]
y_vec = y[np.where(is_train)[0]]

# NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain)
V_mat, w_vec = NNetOneSplit(X_mat, y_vec, 10, 0.1, 10, is_subtrain)
