import tensorflow as tf
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv('data/reuter_train.csv')

#Remove stopwords
stop = set(stopwords.words('english'))
df['text'] = df['text'].str.lower()
df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df= df.sample(frac =1)

#Parameters
embedding_dim = 100
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

#Preprocessing
articles = df['text_without_stopwords']
authors = df['author']

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_authors = authors[0: train_size]

validation_articles = articles[train_size:]
validation_authors = authors[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
vocab_size=len(word_index)

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(authors)

training_author_seq = np.array(author_tokenizer.texts_to_sequences(train_authors))
validation_author_seq = np.array(author_tokenizer.texts_to_sequences(validation_authors))

#Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                             input_length=max_length),
    tf.keras.layers.Conv1D(128, 10, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(51, activation='softmax'),
    tf.keras.layers.Dropout(rate=0.2)
])
model.summary()

#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
num_epochs = 10
CNN = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2)
    
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

#Loading pretrained embeddingds
embeddings_index = {}
f = open('word_embeddings/glove.6B.100d.txt', encoding='utf8')

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
    tf.keras.layers.Conv1D(128, 10, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(51, activation='softmax'),
    tf.keras.layers.Dropout(rate=0.2)
])
model.summary()
                    
#Model optimizer and loss function
optim = tf.keras.optimizers.Adam(lr = 0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
num_epochs = 10
CNN_word_vec = model.fit(train_padded, training_author_seq, epochs=num_epochs, validation_data=(validation_padded, validation_author_seq), verbose=2)
    
#Plot training metrics for model evaluation
plot_graphs(CNN_word_vec, "accuracy")
plot_graphs(CNN_word_vec, "loss")    