# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:51:45 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

raw_data = pd.read_csv(r'C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\KNN Classifier\Assignments\glass.csv')
raw_data.describe()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

data = norm_func(raw_data.iloc[:,:-1])
data.describe()

X = np.array(data.iloc[:,:]) # Predictors 
Y = np.array(raw_data['Type']) # Target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=51)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

pred_train = knn.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_train, pred_train))
