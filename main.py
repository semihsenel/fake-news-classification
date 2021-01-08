# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:17:31 2021

@author: semih
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from RandomForest import RandomForestClassifier
from DecisionTree import DecisionTreeClassifier, Node
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec

word2vec = Word2Vec()
word2vec = Word2Vec.load("word_vectors/vectors.model")

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.stats import entropy
import re
SIZE = 90

#%%
def seperator(vector):
    positive = np.array([])
    negative = np.array([])
    for i in vector:
        if i > 0:
            positive = np.append(positive, i)
        else:
            negative = np.append(negative, i)
    return positive, negative

def predict(title, author, text):
    global my_rf
    df = pd.DataFrame(data = 0, index = [0], columns=data.columns)
    for i in range(df.shape[1]):
        df.iloc[0].iloc[i] = 0
    for i,j in enumerate(sent_tokenize(text)):
        sent = np.array([])
        words = str((re.sub("[^A-Za-z']+", ' ', str(j)).lower())).split(" ")
        v = np.array([])
        for k,l in enumerate(words):
            try:
                v = word2vec.wv[l]
                p_arr, n_arr = seperator(v)
                arr = np.array([np.median(v), v.std(), entropy(p_arr), entropy(n_arr)])
                sent = np.append(sent, arr.copy())
            except:
                pass
        for k in range(1,5):
            try:
                if sent.size < 4:
                    raise Exception()
                p_arr, n_arr = seperator(sent)
                arr = np.array([np.median(sent), sent.std(), entropy(p_arr), entropy(n_arr)])
                df['W{}_{}'.format(j+1,l)].iloc[i] = sent[l-1]
            except:
                pass
    try:
        df[author].iloc[0] = 1
    except:
        pass
    X = df.drop('label',axis=1).values
    return my_rf.predict(list(X)[0])

#%%
# 1 unreliable
# 0 reliable
if __name__ == '__main__':
    data = pd.read_pickle("data/preprocessed.pkl")
    X = data.drop("label",axis = 1).values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    my_rf = joblib.load("data/my_random_forest.joblib")
    rf = joblib.load("data/random_forest.joblib")
    knn = joblib.load("data/knn.joblib")
    gnb = joblib.load("data/gnb.joblib")
    mlp = joblib.load("data/mlp.joblib")
    print("Algorithms Scores")
    print("My Random Forest Implementation : {:.2f}".format(my_rf.score(X_test,y_test)))
    print("Sklearn Random Forest : {:.2f}".format(rf.score(X_test,y_test)))
    print("K-Nearest Neighbors  : {:.2f}".format(knn.score(X_test,y_test)))
    print("Naive Bayes : {:.2f}".format(gnb.score(X_test,y_test)))
    print("Multi Layer Perceptron : {:.2f}".format(mlp.score(X_test,y_test)))

    choice = int(input("Enter 1 to check a news\nEnter 0 to exit\n--> "))
    while choice == 1:
        title = input("Title : ")
        author = input("Author : ")
        text = input("News : ")
        result = predict(title, author, text)
        if result == 0:
            print("The news is reliable")
        else:
            print("The news is unreliable")
        cont = int(input("Enter 1 to continue\nEnter 0 to exit\n--> "))
        if cont == 0:
            choice = 0
#%%

    