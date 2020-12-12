#Character N-Grams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import sklearn
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

#Load dataset
df = pd.read_csv('data/reuter_train.csv')

#Remove stopwords
stop = set(stopwords.words('english'))
df['text'] = df['text'].str.lower()
df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Split Dataset
data_train, data_test, y_train, y_true = \
    train_test_split(df['text_without_stopwords'], df['author'], test_size=0.2, random_state= 42)

#N-grams parameter
ngrams = 3

#Vectorize words into n-grams
ngram_counter = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='char')

X_train = ngram_counter.fit_transform(data_train)
X_test  = ngram_counter.transform(data_test)

#LinearSVC
#Classifier
classifier = LinearSVC(max_iter=2000)

#Train
scores = cross_val_score(classifier, X_train, y_train, cv=10)
cv_score = np.mean(scores)
cv_score

#MultinomialNB
#Classifier
clf = MultinomialNB()
#Train
scores = cross_val_score(clf, X_train, y_train, cv=10)
cv_score = np.mean(scores)
cv_score


#LDA
#Dimensionality reduction
tsvd = TruncatedSVD()
x_tsvd = tsvd.fit_transform(X_train)

#Train
lda = LDA()
scores = cross_val_score(lda, x_tsvd, y_train, cv=10)
cv_score = np.mean(scores)
cv_score
