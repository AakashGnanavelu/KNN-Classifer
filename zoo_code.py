# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:03:57 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\KNN Classifier\Assignments\zoo.csv')
data.describe()

X = np.array(data.loc[:, data.columns != 'animal names'])
Y = np.array(data['type'])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

# error on train data
pred_train = knn.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_train, pred_train))
