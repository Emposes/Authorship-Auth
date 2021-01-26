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

#Tokenize authors
author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(df['author'])

df['author_code'] = np.array(author_tokenizer.texts_to_sequences(df['author']))
df_test['author_code'] = np.array(author_tokenizer.texts_to_sequences(df_test['author']))

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

#Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(256, 10, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Dropout(rate=0.55),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.8),
    tf.keras.layers.Dense(51, activation='softmax')

])
model.summary()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-30)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

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
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(CNN, "accuracy")
plot_graphs(CNN, "loss")    

#Confusion matrix
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
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Conv1D(256, 10, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Dropout(rate=0.6),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.8),
    tf.keras.layers.Dense(51, activation='softmax')
    
])
model.summary()
                    
#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 1e-30)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)

#Train model
num_epochs = 4000
CNN_word_vec = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2, callbacks=[es])
    
#Plot training metrics for model evaluation
plot_graphs(CNN_word_vec, "accuracy")
plot_graphs(CNN_word_vec, "loss")    

#Model Test
y_pred = model.predict(test_padded)
predict_class = np.argmax(y_pred, axis=1)
predict_class = predict_class.tolist()

#Confusion matrix
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
