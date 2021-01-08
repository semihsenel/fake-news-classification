from math import log,sqrt
import random
import numpy as np
import pandas as pd


class Node:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        

class DecisionTreeClassifier:
    def __init__(self, max_depth=-1, random_features=False):
        self.root = None
        self.max_depth = max_depth
        self.features_index = []
        self.random_features = random_features

    def fit(self, rows):
        if len(rows) < 1:
            raise ValueError("Sample size must be largen than 0")
        criterion = self.entropy

        if self.random_features:
            self.features_index = self.choose_random_features(rows[0])
            rows = [self.get_features_subset(row) + [row[-1]] for row in rows]
        
        self.root = self.generate_tree(rows, criterion, self.max_depth)

    def predict(self, X):        
        if self.random_features:
            if not all(i in range(len(X)) for i in self.features_index):
                raise ValueError("Given Features Don't Match")
            X = self.get_features_subset(X)
        return self.classify(X, self.root)
    
    def get_random_features(self, row):
        nb_features = len(row) - 1
        return random.sample(range(nb_features), int(sqrt(nb_features)))
    
    def get_features(self, row):
        return [row[i] for i in self.features_index]
    
    def split_data(self, rows, column, value):
        func = None
        if isinstance(value, int) or isinstance(value, float):
            func = lambda row: row[column] >= value
        else:
            func = lambda row: row[column] == value
        
        set1 = [row for row in rows if func(row)]
        set2 = [row for row in rows if not func(row)]
        return set1, set2

    def unique_number(self, rows):
        results = {}
        for row in rows:
            key = row[len(row) - 1]
            if key not in results:
                results[key] = 0
            results[key] += 1
        return results

    def entropy(self, rows):
        results = self.unique_number(rows)
        value = 0.0
        for key in results.keys():
            p = float(results[key]) / len(rows)
            value = value - p* log(p,2)
        return value
    
    def generate_tree(self, rows, func, depth):
        if len(rows) == 0:
            return Node()
        if depth == 0:
            return Node(results=self.unique_number(rows))
        
        score = func(rows)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(rows[0]) - 1

        for col in range(column_count):
            column_values = {}
            for i in rows:
                column_values[i[col]] = 1
            for i in column_values.keys():
                set1, set2 = self.split_data(rows, col, i)
                p = float(len(set1)) / len(rows)
                gain = score - p * func(set1) - (1 - p) * func(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, i)
                    best_sets = (set1, set2)
        
        if best_gain > 0:
            trueBranch = self.generate_tree(best_sets[0], func, depth - 1)
            falseBranch = self.generate_tree(best_sets[1], func, depth - 1)
            return Node(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
        
        else:
            return Node(results=self.unique_number(rows))

    def classify(self, observation, tree):
        if tree.results is not None:
            return list(tree.results.keys())[0]
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)