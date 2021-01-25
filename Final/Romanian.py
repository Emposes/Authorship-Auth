import tensorflow as tf
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import spacy
from sklearn.model_selection import train_test_split
import sklearn
from scipy import stats
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('main/data/data.csv', sep =";")

df = df[['message', 'from']]

df = df.dropna()

le = LabelEncoder()

df["author_code"] = le.fit_transform(df["from"])

#Split Dataset
data_train, data_test, y_train, y_true = \
    train_test_split(df['message'], df['author_code'], test_size=0.2, random_state= 42)

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

#N-grams parameter
ngrams = 5

#Vectorize words into n-grams
ngram_counter = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='char')

X_train = ngram_counter.fit_transform(data_train)
X_test  = ngram_counter.transform(data_test)
y_train = y_train

#LinearSVC
#Classifier
classifier = LinearSVC(max_iter=1000, dual= False)

#Test
model = classifier.fit(X_train, y_train)
y_test = model.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)

#MultinomialNB
#Classifier
clf = MultinomialNB()

#Test
model = clf.fit(X_train, y_train)
y_test = model.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(0, 9,num=10)
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
tsvd = TruncatedSVD(283)
x_tsvd = tsvd.fit_transform(X_train)

#Test
lda = LDA()
model = lda.fit(x_tsvd, y_train)
test_tsvd = tsvd.transform(X_test)
y_test = model.predict(test_tsvd)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)

#NN
import tensorflow as tf
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import spacy
from sklearn.model_selection import train_test_split
import sklearn
from scipy import stats
from sklearn.metrics import confusion_matrix


#Remove stopwords
stop = set(stopwords.words('romanian'))
df['message'] = df['message'].str.lower()
df['text_without_stopwords'] = df['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


#Split Dataset
data_train, data_test, y_train, y_true = \
    train_test_split(df['text_without_stopwords'], df['author_code'], test_size=0.2, random_state= 42)

#Parameters
vocab_size = 5000
embedding_dim = 100
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

#Preprocessing
articles = data_train
authors = y_train

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_authors = authors[0: train_size]
test_articles = data_test

validation_articles = articles[train_size:]
validation_authors = authors[train_size:]
test_authors = y_true

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

#LSTM
#Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim), merge_mode='concat'),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    #tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(rate=0.8),
    tf.keras.layers.Dense(11, activation='softmax')
])
model.summary()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 5e-5)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)

#Train model
num_epochs = 4000
#History
BiLSTM = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])

#Model Test
y_pred = model.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Plot training metrics for model evaluation
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(BiLSTM, "accuracy")
plot_graphs(BiLSTM, "loss")    

#Confusion matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#CNN
#Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(256, 10, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Dropout(rate=0.55),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.8),
    tf.keras.layers.Dense(11, activation='softmax')

])
model.summary()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-30)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)

#Train model
num_epochs = 4000
CNN = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])
    
#Model Test
y_pred = model.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Plot training metrics for model evaluation
plot_graphs(CNN, "accuracy")
plot_graphs(CNN, "loss")    

#Confusion matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#CNN + LSTM
#Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 10, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=51),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(11, activation='softmax')
])
model.summary()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-30)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)

#Train model
num_epochs = 4000
CNN_LSTM = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])

#Model Test
y_pred = model.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Plot training metrics for model evaluation
plot_graphs(CNN_LSTM, "accuracy")
plot_graphs(CNN_LSTM, "loss")    

#Confusion matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#Attention Model
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
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
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
    model.add(keras.layers.Dense(11, activation = 'softmax'))

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

#Model Test
y_pred = modelH.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Plot training metrics for model evaluation
plot_graphs(LSTM_AttnH, "accuracy")
plot_graphs(LSTM_AttnH, "loss")    

#Confusion matrix
confmat=confusion_matrix(test_author_seq, predict_class)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(test_author_seq, predict_class)

#BERT
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn 
from tensorflow import keras

# Select required columns
data = df[['text_without_stopwords', 'from']]
data['author_label'] = pd.Categorical(data['from'])
data['author'] = data['author_label'].cat.codes

# Split into train and test - stratify over Issue
data, data_test = train_test_split(data, test_size = 0.2, random_state = 42)

# Name of the BERT model to use
model_name = 'bert-base-uncased'
# Max length of tokens
max_length = 100
# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)

# Load the MainLayer
bert = transformer_model.layers[0]
# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}
# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Then build your model output
author = Dense(units=len(data.author.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='author')(pooled_output)
outputs = {'author': author}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiClass')
# Take a look at the model
model.summary()

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss = {'author': CategoricalCrossentropy(from_logits = True)}
metric = {'author': CategoricalAccuracy('accuracy')}
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)
# Ready output data for the model
y_author = to_categorical(data['author'])
# Tokenize the input (takes some time)
x = tokenizer(
    text=data['text_without_stopwords'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)

# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'author': y_author},
    validation_split=0.2,
    batch_size=64,
    verbose=2,
    epochs=100,
    callbacks=[es])

#Save model
model.save('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/Final/Romana/BERT')

#Load Moel
model = keras.models.load_model('C:/Users/Win/Desktop/Doctorat/Poli/Proiect/Final/Romana/BERT')

# Ready test data
test_y_author = to_categorical(data_test['author'])
test_x = tokenizer(
    text=data_test['text_without_stopwords'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)
# Run evaluation
model_eval = model.evaluate(
    x={'input_ids': test_x['input_ids']},
    y={'author': test_y_author})

#Plot training metrics for model evaluation
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")    

#Confusion matrix
predict_test = model.predict(test_x['input_ids'])
aba = list(predict_test.values())
datab = pd.DataFrame(aba[0])
an_array = np.argmax([datab],axis=2)
an_array = np.transpose(an_array)

confmat = confusion_matrix(data_test['author'], an_array)
ticks = np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Test Accuracy
sklearn.metrics.accuracy_score(data_test['author'], an_array)

#Inference
import re

def predict_author(text, classifier, vectorizer):

    #text = preprocess_text(text)
    text = [text]
    vectorized_text = vectorizer(
       text=text,
       add_special_tokens=True,
       max_length=max_length,
       truncation=True,
       padding=True, 
       return_tensors='tf',
       return_token_type_ids = False,
       return_attention_mask = False,
       verbose = False)
    result = classifier.predict(vectorized_text['input_ids'])
    result = list(result.values())
    result = pd.DataFrame(result[0])
    result = np.argmax([result],axis=2)
    #result = result+1
    #result = author_tokenizer.sequences_to_texts(result)
    return {'author': result}

df2 = df[df['text_without_stopwords'].apply(len)>600]

def get_samples():
    samples = {}
    for auth in df.author_code.unique():
        samples[auth] = df2.text_without_stopwords[df.author_code==auth].tolist()[:3]
    return samples

val_samples = get_samples()


#title = input("Enter a news title to classify: ")
classifier = model

for truth, sample_group in val_samples.items():
    print(f"True Category: {truth}")
    print("="*30)
    for sample in sample_group:
        prediction = predict_author(sample, classifier, 
                                      tokenizer)
        print("Prediction: {} ".format(prediction['author']))
        print("\t + Sample: {}".format(sample[:100]))
    print("-"*30 + "\n")
    
#String Kernels
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

xs = data_train.tolist()
ys = y_train.tolist()

us = data_test.tolist()
vs = y_true.tolist()
#SVC
train_kernel = computeKernelMatrix2(xs, xs)
svc = SVC(kernel='precomputed')
svc.fit(train_kernel, ys)


test_kernel = computeKernelMatrix2(us, xs)
y_test = svc.predict(test_kernel)


#Dimensionality reduction
tsvd = TruncatedSVD(50)
x_tsvd = tsvd.fit_transform(train_kernel)
svc = SVC(kernel='precomputed')
svc.fit(x_tsvd, ys)

#Confusion Matrix
confmat=confusion_matrix(vs, y_test)
ticks=np.linspace(0, 9,num=10)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(vs, y_test)

