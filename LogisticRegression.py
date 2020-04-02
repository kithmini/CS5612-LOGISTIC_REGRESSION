#!/usr/bin/env python
# coding: utf-8

# In[3]:


print('##### Logistic Regression on Wisconsin Breast Cancer Data #####')

print('----- Importing required libraries & modules-----')

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import scipy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


# In[4]:


print('----- Importing dataset -----')
data = pd.read_csv('bcwd.csv', header=None)

data.columns = ['sample_code_number','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
feature_columns = ['clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']



print ('Imported Rows, Columns - ', data.shape)
print ('Data Head :')
data.head()


# In[5]:


missingRemovedData =  data[data['bare_nuclei'] != '?'] # remove rows with missing data
missingReplacedData = data.replace(to_replace='?', value='1') # replace missing data with 1

print ('Missing Removed Rows, Columns - ', missingRemovedData.shape)
print ('Missing Replaced Rows, Columns - ', missingReplacedData.shape)

X = missingRemovedData[feature_columns]
y = missingRemovedData['class']

X1 = missingReplacedData[feature_columns]
y1 = missingReplacedData['class']

# split X and y into training and teting sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100)
model = LogisticRegression()
model.fit(X_train,y_train)
y = model.score(X_test, y_test)

print('\n----- Removing Missing Values -----')
print("Accuracy: %.2f%%" % (y*100.0))
print("Confusion Matrix:")
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[6]:


# split X and y into training and teting sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.35, random_state=100)
model1 = LogisticRegression()
model1.fit(X1_train,y1_train)
y1 = model1.score(X1_test, y1_test)

print('\n----- Replacing Missing Values -----')
print("Accuracy: %.2f%%" % (y1*100.0))
print("Confusion Matrix:")
y1_pred = model1.predict(X1_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y1_test, y1_pred)
print(confusion_matrix)

