#%%
# Importing Data
import numpy as np
import pandas as pd

data = pd.read_csv("data/news.csv")
data.set_index('id', inplace=True)
data = data.fillna('')
#%%
# Outlier Detection (Low Sensivity)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
vals = np.array([])
for i,j in enumerate(data['text']):
    vals = np.append(vals, len(sent_tokenize(j))) # textlerdeki cümle sayılarını tuttum
print("Mean :",vals.mean())
print("Max :",vals.max())
print("Std :",vals.std())

Q1 = np.quantile(vals,0.25)
Q3 = np.quantile(vals,0.75)
iqr_max = (Q3-Q1)*2.5
print("IQR :",iqr_max)
# cümle sayısı olarak outlier değerleri bulup çıkarttım
indexes = []
for i,j in enumerate(data['text']):
    if len(sent_tokenize(j)) > 54:
        indexes.append(i)
print("Outlier Size :",len(indexes))

data = data.drop(data.index[indexes])
labels = data["label"].values.copy()
data.drop("label", axis=1, inplace=True)
#%%
# combine all text and all tites to use word2vec
text = ""
for i,j in zip(data['title'], data['text']):
    text += i + "." + j
#%%
# tokenizing to use word2vec
import spacy
import re
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
text = sent_tokenize(text)
text = pd.DataFrame(text, columns=['sentences'])
nlp = spacy.load('en', disable=['parser'])

# text stringini cümlelere ayırarak bir dataframe oluşturdum
def tokenize(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
# oluşturduğum dataframei temizledim
data_filtered = (re.sub("[^A-Za-z']+", " ", str(row)).lower() for row in text['sentences'])
txt = [tokenize(doc) for doc in nlp.pipe(data_filtered, batch_size=1000, n_threads=-1)]
df_cleaned = pd.DataFrame({'cleaned':txt})
df_cleaned = df_cleaned.dropna().drop_duplicates()
                                                    
sentence = [row.split() for row in df_cleaned['cleaned']]
phrases = Phrases(sentence, min_count=3, progress_per=10000)
sentences = phrases[sentence]

# kelime frekanslarını buldum
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1

print(list(word_freq.items()))
#%%
# Building Word2Vec model
from gensim.models import Word2Vec
word2vec = Word2Vec(min_count=2, window=2, size=300, sample=6e-5, alpha=0.03, 
                     min_alpha=0.0007, negative=20, workers=1)
word2vec.build_vocab(sentences, progress_per=10000)
# word2vec ile kelimeleri vektörlere çevirdim
word2vec.train(sentences, total_examples=word2vec.corpus_count, epochs=30, report_delay=1)
word2vec.init_sims(replace=True)
#%%
# Adding new columns
SIZE = int(iqr_max)
for i in range(1,SIZE):
    for j in range(1,5):
        data["W{}_{}".format(i,j)] = 0
        data["W{}_{}".format(i,j)] = data["W{}_{}".format(i,j)].astype(object)
# dataframede her cümle için 4 sütunluk yer açtım   
#%%
# Converting words to vectors
# Minimize vectors
# Combine word vectors for each sentence
# Minimize sentence vectors
import scipy 
from scipy.stats import entropy

def seperator(vector):
    # pozitif ve negatif sayıların ayrı ayrı entropisini hesaplamak için
    # diziyi ayırdım
    positive = np.array([])
    negative = np.array([])
    for i in vector:
        if i > 0:
            positive = np.append(positive, i)
        else:
            negative = np.append(negative, i)
    return positive, negative

for i in range(len(data)):
    txt = data['text'].iloc[i]
    for j,k in enumerate(sent_tokenize(txt)):
        sent = np.array([])
        words = str((re.sub("[^A-Za-z']+", ' ', str(k)).lower())).split(" ")
        v = np.array([])
        for l,m in enumerate(words):
            try:
                # her bir kelime vektörü için 4 elemanlı bir array oluşturdum
                v = word2vec.wv[m]
                p_arr, n_arr = seperator(v)
                arr = np.array([np.median(v), v.std(), entropy(p_arr), entropy(n_arr)])
                sent = np.append(sent, arr.copy())
            except:
                pass
        for l in range(1,5):
            try:
                if sent.size < 4:
                    raise Exception()
                p_arr, n_arr = seperator(sent)
                # her bir cümle için 4 elemanlı kelime arraylerini birleştirip yeni bir 
                # 4 elemanli array oluşturdum ve sütunlara yerleştirdim
                arr = np.array([np.median(sent), sent.std(), entropy(p_arr), entropy(n_arr)])
                data['W{}_{}'.format(j+1,l)].iloc[i] = sent[l-1]
            except:
                if j < SIZE-1:
                    data['W{}_{}'.format(j+1,l)].iloc[i] = 0

#%%
# Dropping old columns and OneHodEncoding for authors
data.drop("title", axis = 1, inplace=True)
data.drop("text", axis = 1, inplace=True)

dummies = pd.get_dummies(data["author"])
data = pd.concat([data,dummies], axis = 1)
# yazarlar için OneHotEncoding uyguladım
data["label"] = labels.copy()
data.drop("author", axis = 1, inplace = True)
#%%
# Saving preprocessed data
data.to_pickle("data/preprocessed.pkl", protocol=4)
#%%



































