# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:08:42 2021

@author: semih
"""
import numpy as np
import pandas as pd
import scipy.stats as st
from DecisionTree import DecisionTreeClassifier, Node
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from math import log,sqrt
import random

class RandomForestClassifier:
    def __init__(self, nb_trees, nb_samples = None, n_estimators = 100, max_depth = 5, max_workers=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_workers = max_workers
    
    def fit(self, X, y):
        if not self.nb_samples:
            self.nb_samples = int(sqrt(X.shape[0]))
        data = list(np.concatenate((X, np.array([y]).T), axis=1))
        with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
            rand = map(lambda x: [x, random.sample(data, self.nb_samples)], range(self.nb_trees))
            self.trees = list(executor.map(self.train_tree, rand))
        print("Training Completed")
        
    def train_tree(self, data):
        if (data[0]+1) % 25 == 0:
            print("Training Tree {}".format(data[0] + 1))
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data[1])
        return tree
    
    def predict(self, X):
        pred = []
        for tree in self.trees:
            pred.append(tree.predict(X))
        
        return max(set(pred), key=pred.count)
    
    def score(self, X, y):
        features = list(X)
        values = list(y)
        t = 0
        for i,j in zip(features, values):
            prediction = self.predict(i)
            if prediction == j:
                t += 1
        return t / len(values)
