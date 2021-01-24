#Attention
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer

from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.backend as K
from keras.callbacks import *

import tensorflow as tf
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import EarlyStopping
import spacy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
from scipy import stats

#Load dataset
df = pd.read_csv('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/GitHub/data/reuter_train.csv')

df_test = pd.read_csv('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/GitHub/data/reuter_test.csv')

#Split into sentences
stop = set(stopwords.words('english'))
df['text'] = df['text'].str.lower()
df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

nlp = spacy.load("en_core_web_sm")
df["text_without_stopwords"] = df["text_without_stopwords"].apply(lambda x: [sent.text for sent in nlp(x).sents])
df = df.explode('text_without_stopwords')
df_test['text_without_stopwords'] = df_test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenize authors
author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(df['author'])
df['author_code'] = np.array(author_tokenizer.texts_to_sequences(df['author']))
df_test['author_code'] = np.array(author_tokenizer.texts_to_sequences(df_test['author']))

#For Inference
new = df[['author', 'author_code']].copy()
new = new.drop_duplicates()
new = new.set_index('author_code')
new = new.to_numpy()

#Parameters
vocab_size = 5000
embedding_dim = 100
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

#Preprocessing
articles = df['text_without_stopwords']
authors = df['author_code']

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_authors = authors[0: train_size]
test_articles = df_test['text_without_stopwords']

validation_articles = articles[train_size:]
validation_authors = authors[train_size:]
test_authors = df_test['author_code']

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
vocab_size=len(word_index)

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_articles)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_author_seq = train_authors
validation_author_seq = validation_authors
test_author_seq = test_authors

#Attention Model
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(keras.layers.Layer):

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

#Build model
def LSTM_Attn():
    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size, embedding_dim))
    model.add(keras.layers.LSTM(embedding_dim, return_sequences=True))
    #model.add(keras.layers.Dropout(rate = 0.4))
    model.add(AttentionWithContext())
    #model.add(keras.layers.Dropout(rate = 0.4))
    model.add(keras.layers.Dense(embedding_dim, activation = 'relu'))
    model.add(keras.layers.Dropout(rate = 0.8))
    model.add(keras.layers.Dense(51, activation = 'softmax'))

    return model

#Model History
modelH = LSTM_Attn()
modelH.summary()

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-3)
modelH.compile(loss='sparse_categorical_crossentropy', optimizer= optim , metrics=['accuracy'])

#Train model
num_epochs = 4000
LSTM_AttnH = modelH.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])

#Plot training metrics for model evaluation
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(LSTM_AttnH, "accuracy")
plot_graphs(LSTM_AttnH, "loss")  

#Model Test
y_pred = modelH.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Confusion Matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#Loading pretrained embeddingds
embeddings_index = {}
f = open('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/Alfa/glove.6B.100d.txt', encoding='utf8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embeddings_matrix = np.zeros((len(word_index) + 1, max_length))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector
                    
#Build model with preprocessed word emebddigns

def LSTM_Attn_Glove():
    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False))
    model.add(keras.layers.LSTM(embedding_dim, return_sequences=True))
    #model.add(keras.layers.Dropout(rate = 0.4))
    model.add(AttentionWithContext())
    #model.add(keras.layers.Dropout(rate = 0.4))
    model.add(keras.layers.Dense(embedding_dim, activation = 'relu'))
    model.add(keras.layers.Dropout(rate = 0.8))
    model.add(keras.layers.Dense(51, activation = 'softmax'))

    return model

#Model History
modelH_G = LSTM_Attn_Glove()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-3)
modelH_G.compile(loss='sparse_categorical_crossentropy', optimizer= optim, metrics=['accuracy'])

#Train model
num_epochs = 4000
LSTM_word_vec = modelH_G.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])
    
#Plot training metrics for model evaluation
plot_graphs(LSTM_word_vec, "accuracy")
plot_graphs(LSTM_word_vec, "loss")    

#Model Test
y_pred = modelH_G.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Confusion Matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(1, 50,num=50)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#Inference
import re

# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def predict_author(text, classifier, vectorizer):

    text = preprocess_text(text)
    vectorized_text = vectorizer(text)
    text_padded = pad_sequences(vectorized_text, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = classifier.predict(text_padded)
    predicted_author = result.mean(axis=0)
    indices= np.argmax(predicted_author, axis=0)
    probability_values = np.max(predicted_author)
    predicted_author = new[indices]
    
    return {'author': predicted_author, 
            'probability': probability_values}

def get_samples():
    samples = {}
    for auth in df.author.unique():
        samples[auth] = df.text[df.author==auth].tolist()[:5]
    return samples

val_samples = get_samples()

#title = input("Enter a news title to classify: ")
classifier = model_G

for truth, sample_group in val_samples.items():
    print(f"True Category: {truth}")
    print("="*30)
    for sample in sample_group:
        prediction = predict_author(sample, classifier, 
                                      tokenizer.texts_to_sequences)
        print("Prediction: {} (p={:0.2f})".format(prediction['author'],
                                                 prediction['probability']))
        print("\t + Sample: {}".format(sample[:100]))
    print("-"*30 + "\n")


    
    