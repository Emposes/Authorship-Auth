#BERT
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn 
from tensorflow import keras

# Import data from csv
df = pd.read_csv('main/data/reuter_train.csv')

df_test = pd.read_csv('main/data/reuter_test.csv')

df_test= df_test.sample(frac =1, random_state=42)

df= df.sample(frac =1, random_state=42)
#Tokenize authors
author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(df['author'])
df['author_code'] = np.array(author_tokenizer.texts_to_sequences(df['author']))
df_test['author_code'] = np.array(author_tokenizer.texts_to_sequences(df_test['author']))

# Select required columns
data = df[['text', 'author']]
data['author_label'] = pd.Categorical(data['author'])
data['author'] = data['author_label'].cat.codes

data_test = df_test[['text', 'author']]
data_test['author_label'] = pd.Categorical(data_test['author'])
data_test['author'] = data_test['author_label'].cat.codes

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
    text=data['text'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

#Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)

# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'author': y_author},
    validation_split=0.2,
    batch_size=64,
    verbose=2,
    epochs=100,
    callbacks=[es])

# Ready test data
test_y_author = to_categorical(data_test['author'])
test_x = tokenizer(
    text=data_test['text'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)
# Run evaluation
model_eval = model1.evaluate(
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
predict_test = model1.predict(test_x['input_ids'])
aba = list(predict_test.values())
datab = pd.DataFrame(aba[0])
an_array = np.argmax([datab],axis=2)
an_array = np.transpose(an_array)

confmat = confusion_matrix(data_test['author'], an_array)
ticks = np.linspace(1, 50,num=50)
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
    result = result+1
    result = author_tokenizer.sequences_to_texts(result)
    return {'author': result}

df2 = df[df['text'].apply(len)>600]

def get_samples():
    samples = {}
    for auth in df.author.unique():
        samples[auth] = df2.text[df.author==auth].tolist()[:10]
    return samples

val_samples = get_samples()

classifier = model1

for truth, sample_group in val_samples.items():
    print(f"True Category: {truth}")
    print("="*30)
    for sample in sample_group:
        prediction = predict_author(sample, classifier, 
                                      tokenizer)
        print("Prediction: {} ".format(prediction['author']))
        print("\t + Sample: {}".format(sample[:100]))
    print("-"*30 + "\n")
 
