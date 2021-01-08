# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:09:38 2021

@author: semih
"""
#%%
# Importing data
import pandas as pd
import numpy as np
import joblib

data = pd.read_pickle("data/preprocessed.pkl")
#%%
# Splitting data
from sklearn.model_selection import train_test_split
X = data.drop("label",axis = 1).values
y = data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
#%%
# My Algorithm 
from RandomForest import RandomForestClassifier
model = RandomForestClassifier(nb_trees=200, max_depth=50, n_estimators=300, max_workers=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

joblib.dump(model, "data/my_random_forest.joblib")

#%%
from sklearn.ensemble import RandomForestClassifier as RFC
rf = RFC(max_depth=50, n_estimators=300, criterion='entropy', verbose=0, max_features="sqrt")
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

joblib.dump(rf, "data/random_forest.joblib")

#%%
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))

joblib.dump(neigh, "data/knn.joblib")

#%%

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(gnb.score(X_test, y_test))

joblib.dump(gnb, "data/gnb.joblib")

#%%
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0, max_iter=300)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

joblib.dump(mlp, "data/mlp.joblib")


#%%