# Fake News Classification

### Requirements
```py
pip install pandas==1.1.3
pip install numpy==1.19.2
pip install sklearn==0.0
pip install scipy==1.5.2
pip install joblib==0.17.0
pip install gensim==3.8.3
pip install nltk==3.5
```
Data : https://www.kaggle.com/c/fake-news/data

### Preprocess
I have used word2vec to convert text data to numeric values. In this project, I obtained an array shaped (300,1) for each word and minimized each array to (4,1) by using median, mean, entropy of positive values and entropy of negative values of the array. Then, I concatenated each array of a sentence and also minimized these arrays. In the end, I obtained an array shaped (4,1) for each sentence and placed 4 values to 4 columns of dataframe.

### Train
I used an implementation of Random Forest to train. I used sklearn libraries to compare accuracy.
**Results:**
Implementation of Random Forest : 0.72
Sklearn Random Forest : 0.83
Sklearn K-Nearest Neighbors : 0.79
Sklearn Naive Bayes : 0.84
Sklearn Multi Layer Perceptron : 0.96
