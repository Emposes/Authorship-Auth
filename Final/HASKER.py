#Character N-Grams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import string
from sklearn.preprocessing import normalize
from sklearn.kernel_ridge import KernelRidge
import time
from nltk.corpus import stopwords
from sklearn.svm import SVC

#Load dataset
df = pd.read_csv('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/GitHub/data/reuter_train.csv')
df_test = pd.read_csv('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/GitHub/data/reuter_test.csv')

df_test= df_test.sample(frac =1, random_state=42)

df= df.sample(frac =1, random_state=42)
#Sentence parsing
nlp = spacy.load("en_core_web_sm")
df["text"] = df["text"].apply(lambda x: [sent.text for sent in nlp(x).sents])
df = df.explode('text')

#Remove stopwords
stop = set(stopwords.words('english'))
df['text'] = df['text'].str.lower()
df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df_test['text'] = df_test['text'].str.lower()
df_test['text_without_stopwords'] = df_test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenize authors
author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(df['author'])
df['author_code'] = np.array(author_tokenizer.texts_to_sequences(df['author']))
df_test['author_code'] = np.array(author_tokenizer.texts_to_sequences(df_test['author']))

xs = df['text_without_stopwords'].tolist()
ys = df['author_code'].tolist()

us = df_test['text_without_stopwords'].tolist()
vs = df_test['author_code'].tolist()

def computeKernelMatrix2(X,Y):
    start = time.time()
    K = np.zeros((len(X),len(Y)))
    count = 0

  
    for i in range(len(X)):
        ngrams = {}
        for l in range(len(X[i])):
            if l + 3 <= len(X[i]):
                
                ngram = X[i][l: l + 3]
                if ngram in ngrams:
                    count = ngrams[ngram]
                else:
                    ngrams[ngram] = 0
                    count = ngrams[ngram]

                if count != 0:
                    ngrams[ngram] = count + 1
                else:
                    ngrams[ngram] = 1

        
        
        for j in range(i,len(Y)):
            K[i][j]=0
            for l in range(len(Y[j])):
                if l + 3 <= len(Y[j]):
                    ngram = Y[j][l: l + 3]
                    if ngram in ngrams:
                        count = ngrams[ngram]
                    else:
                        ngrams[ngram] = 0
                        count = ngrams[ngram]
                if count != 0 and count > 0:
                    K[i][j] += count
                    
            K[j][i] = K[i][j]
            
          
        
        print(i/len(X)*100)
    print ("Calculated in {} seconds".format(time.time() - start))
    return K

#SVC
train_kernel = computeKernelMatrix2(xs, xs)
svc = SVC(kernel='precomputed')
svc.fit(train_kernel, ys)

pd.DataFrame(train_kernel).to_csv("C:/Users/Win/Desktop/Doctorat/Poli/Proiect/Final/StringKernels/train.csv")

test_kernel = computeKernelMatrix2(us, xs)
y_test = svc.predict(test_kernel)

pd.DataFrame(test_kernel).to_csv("C:/Users/Win/Desktop/Doctorat/Poli/Proiect/Final/StringKernels/test.csv")

#Dimensionality reduction
tsvd = TruncatedSVD(50)
x_tsvd = tsvd.fit_transform(train_kernel)
svc = SVC(kernel='precomputed')
svc.fit(x_tsvd, ys)

#Confusion Matrix
confmat=confusion_matrix(vs, y_test)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(vs, y_test)

