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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Load dataset
df = pd.read_csv('main/data/reuter_train.csv')

df_test = pd.read_csv('main/data/reuter_test.csv')

df_test= df_test.sample(frac =1, random_state=42)

df= df.sample(frac =1, random_state=42)
#Remove stopwords
stop = set(stopwords.words('english'))
df['text'] = df['text'].str.lower()
df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df_test['text'] = df_test['text'].str.lower()
df_test['text_without_stopwords'] = df_test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Split Dataset
data_train, data_test, y_train, y_true = \
    train_test_split(df['text_without_stopwords'], df['author'], test_size=0.2, random_state= 42)

#N-grams parameter
ngrams = 5

#Vectorize words into n-grams
ngram_counter = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='char')

X_train = ngram_counter.fit_transform(df['text_without_stopwords'])
X_test  = ngram_counter.transform(df_test['text_without_stopwords'])
y_train = df['author'] 
y_true = df_test['author']
#LinearSVC
#Classifier
classifier = LinearSVC(max_iter=1000, dual= False)

#Train
scores = cross_val_score(classifier, X_train, y_train, cv=10)
cv_score = np.mean(scores)
cv_score

#Test
model = classifier.fit(X_train, y_train)
y_test = model.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)

#Inference 
import re

def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def predict_author(text, classifier, vectorizer):

    text = preprocess_text(text)
    text = [text]
    vectorized_text = vectorizer(text)
    result = classifier.predict(vectorized_text)
    
    return {'author': result}

def get_samples():
    samples = {}
    for auth in df.author.unique():
        samples[auth] = df.text[df.author==auth].tolist()[:5]
    return samples

val_samples = get_samples()

#title = input("Enter a news title to classify: ")
classifier = model

for truth, sample_group in val_samples.items():
    print(f"True Category: {truth}")
    print("="*30)
    for sample in sample_group:
        prediction = predict_author(sample, classifier, 
                                      ngram_counter.transform)
        print("Prediction: {} ".format(prediction['author']))
        print("\t + Sample: {}".format(sample[:100]))
    print("-"*30 + "\n")
    
    
#MultinomialNB
#Classifier
clf = MultinomialNB()
#Train
scores = cross_val_score(clf, X_train, y_train, cv=10)
cv_score = np.mean(scores)
cv_score

#Test
model = clf.fit(X_train, y_train)
y_test = model.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)


#LDA
#Dimensionality reduction
tsvd = TruncatedSVD(520)
x_tsvd = tsvd.fit_transform(X_train)

#Train
lda = LDA()
scores = cross_val_score(lda, x_tsvd, y_train, cv=10)
cv_score = np.mean(scores)
cv_score

#Test
model = lda.fit(x_tsvd, y_train)
test_tsvd = tsvd.transform(X_test)
y_test = model.predict(test_tsvd)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)
