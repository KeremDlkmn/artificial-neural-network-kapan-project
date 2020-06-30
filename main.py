import pandas as pd
import numpy as np
from functions import plotly_dataset_functions, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Read a dataset
dataset = pd.read_csv('dataset/spam.csv', delimiter=',', encoding='latin-1')

# Drop Unnamed columns
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Plots spam and ham values
plotly_dataset_functions.plot_dataset_columns(dataset.v1, 'Labels', 'Number Of Spam & Ham')

# Preparing data
X_data = dataset.v2
Y_data = dataset.v1

# Converts output values to numerical values.
label_encoder = LabelEncoder()
Y_data        = label_encoder.fit_transform(Y_data)
Y_data        = Y_data.reshape(-1, 1)

# We divided the Dataset into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.15)

# Magic numbers. It was arbitrarily chosen,
# But a method can be developed by analyzing the lengths of sentences. workable
max_words = 1000
max_len = 150

# Tokenizer: used to convert text given as input into matrices
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces
# it with its corresponding integer value from the word_index dictionary
sequences = tokenizer.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# Create LSTM model
model = models.rnn_lstm(max_len, max_words)

# Create model summary
model.summary()

# Model compiled
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Model started to be educated
model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

test_sequences = tokenizer.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

# Evaluation of the model with test values
accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

# Single Predict
spam_texts = ["WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]

spam_txts = tokenizer.texts_to_sequences(spam_texts)
spam_txts = sequence.pad_sequences(spam_txts, maxlen=max_len)
spam_preds = model.predict(spam_txts)
print("Spam", spam_preds)

# Single Predict
secure_texts = ["As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]

secure_txts = tokenizer.texts_to_sequences(secure_texts)
secure_txts = sequence.pad_sequences(secure_txts, maxlen=max_len)
secure_preds = model.predict(secure_txts)
print("Secure", secure_preds)