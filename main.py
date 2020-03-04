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
    X_validation = np.array([[]])
    n_col = X_mat.shape[1]
    # V_mat contains all of the small w values for each hidden unite, layer 1
    # w_vec contains all of the small w values for y_hat, layer 2
    V_mat = stats.zscore(np.random.random_sample(n_col, n_hidden_units)) / 10
    w_vec = stats.zscore(np.random.random_sample(n_col)) / 10
    
    for i in range(is_subtrain.shape[0]):
        if(is_subtrain[i] == "TRUE"):
            X_subtrain.append(np.copy(X_mat[i]))
        if(is_subtrain[i] == "FALSE"):
            X_validation.append(np.copy(X_mat[i]))

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
            grad_w_list = Back_Propagation(h_list, weight_list)
            # update the theta for each weight
            for i in range(len(weight_list))
                weight_list[i] = weight_list[i] - step_size * grad_w_list[i]
            weight_list_X.append(weight_list)

        ## compute the logistic loss for subtraihttps://forum.handsontable.com/t/adding-dictionary-map-array-as-data/4199/3n and validaiton
        # get prediction for subtrain and validaiton
        
        

    #return h_list
    #return V_mat, w_vec

# logistic loss function
def LogisticLoss(pred, label)
    value = log(1 + exp(-label * pred)
    return value

# forward propagation function
def ForwardPropagation(X, weight_list)
    h_list = list()
    h_list.append(X)# is it the correct way to append a numpy array into a list? or use np.copy(X)
    for i in range(len(weight_list))
        a_vec = np.matmul(X, weight_list[i])
        if(i == len(weight_list))
            h_list.append(a_vec)# is it the correct way to append a numpy array into a list?
        else
            h_vec = 1/(1+exp(-a_vec))
            h_list.append(h_vec)# is it the correct way to append a numpy array into a list?

    return h_list

# back progration function
def Back_Propagation(h_list, w_list):
    grad_w_list = list()
    for l in range(2, 1):
        if(l == 2):
            grad_a = -1 * y_tilde / (1 + exp(np.matmul(h_list[l])))
        else:
            grad_h = np.matmul(w_list[l].T, grad_a)
            grad_a = grad_h * h_list[l] * (1 - h_list[l])
        grad_w = np.matmul(grad_a[l], h_list[l - 1].T)
        grad_w_list.append(grad_w)# is it the correct way to append a numpy array into a list?

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
is_train = np.random.randint(0, 5, X.shape[0]) # 80% train, 20% test (from whole dataset)
is_train = (is_train < 4) # convert to boolean

#Next create a variable is.subtrain (logical vector with size equal to the number of observations in the train set).
is_subtrain = np.random.randint(0, 5, is_train[is_train == True].shape[0]) # 60% subtrain, 40% validation (from training set)
is_subtrain = (is_subtrain < 3) # convert to boolean

# get training set
X_mat = X[np.where(is_train)[0]]
y_vec = y[np.where(is_train)[0]]

# NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain)
# import pdb; pdb.set_trace()
V_mat, w_vec = NNetOneSplit(X_mat, y_vec, 10, 0.1, 10, is_subtrain)
