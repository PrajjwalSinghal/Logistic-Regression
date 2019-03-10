#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:28:04 2019

@author: prajjwalsinghal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Importing the dataset

dataset = pd.read_csv('breast-cancer-wisconsin.csv')

#Modifying the values 
dataset.loc[dataset.Class == 2, 'Class'] = 'benign'
dataset.loc[dataset.Class == 4, 'Class'] = 'malignant'

#Removing the values that are missing some data
dataset = dataset.dropna()

#removing the id column
dataset = dataset.drop('Id', 1)

# Converting the data to integers
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].astype('int64')

# Modifying the values in string to int
dataset.loc[dataset.Class == 'benign', 'Class'] = 0
dataset.loc[dataset.Class == 'malignant', 'Class'] = 1

# Seprating data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(dataset, test_size = 0.3)
X_train.Class.value_counts()

# separating the majority and minority class
df_majority = X_train[X_train.Class == 0]
df_minority = X_train[X_train.Class == 1]

# Downsample majority
df_majority_downsampled = resample(df_majority,
                                   replace = False,
                                   n_samples = 168,
                                   random_state = 123)
# Combine majority and minority
X_train_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Checking the results
X_train_downsampled.Class.value_counts()

# Separating input features and tareget variable
Y = X_train_downsampled.Class
X = X_train_downsampled.drop('Class', axis = 1)
Y_test = X_test.Class
X_test = X_test.drop('Class', axis = 1)

# Training the model
clf = LogisticRegression().fit(X,Y)

#Predicting on test set
Y_Pred = clf.predict(X_test)

print(np.unique( Y_Pred ))

print(accuracy_score(Y_test, Y_Pred))

# accuracy = 0.975609756097561












