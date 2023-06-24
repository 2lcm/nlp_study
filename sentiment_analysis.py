import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
print(len(train_dataset), len(test_dataset))

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for sent, label in train_dataset:
    train_sentences.append(str(sent.numpy()))
    train_labels.append(label.numpy())

for sent, label in test_dataset:
    test_sentences.append(str(sent.numpy()))
    test_labels.append(label.numpy())

vocab_size = 10000
max_length = 150

# 토큰
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(train_sentences)

# 문장 토큰화
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# 데이터 길이 맞추기
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating='post', padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating='post', padding='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 모델
model = Sequential([
    Embedding(vocab_size+1, 64),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
print(model.summary())

try:
    model = keras.models.load_model('models/sentiment_analysis')
except:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_padded, train_labels, epochs=30, batch_size=128, validation_data=(test_padded, test_labels), verbose=1)
    model.save('models/sentiment_analysis')

# test
sample_text = ['The movie was terrible.', 'The movie as fantastic. I would recommend the movie', 'The animation and graphics were out of this world']

sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_seq, maxlen=max_length, padding='post', truncating='post')

ret = model.predict(sample_padded)
print(ret)