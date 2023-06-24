import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Activation
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split

# 데이터 전처리
file_path = "data/train.txt"

tagged_sentences = []
sentence = []

with open(file_path, 'r') as f:
    lines = f.read().strip().split("\n")

for i, line in enumerate(lines):
    if len(line) == 0 or line.startswith("-DOCSTART"):
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
    else:
        word, pos_tag, chunk_tag, ner = line.split()
        word = word.lower()
        sentence.append((word, ner))
if len(sentence) > 0:
    tagged_sentences.append(sentence)
    sentence = []

print(tagged_sentences[0])

inputs, labels = [], []

for pairs in tagged_sentences:
    words, tags = zip(*pairs)
    inputs.append(list(words))
    labels.append(list(tags))

MAX_LENGTH = 60
MAX_WORDS = 4000

tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs)

train_sentences, test_sentences, train_tags, test_tags = train_test_split(inputs, labels, test_size=0.2, random_state=0)

entity_tokenizer = Tokenizer(oov_token='OOV')
entity_tokenizer.fit_on_texts(train_sentences)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(labels)

vocab_size = len(entity_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

x_train = entity_tokenizer.texts_to_sequences(train_sentences)
y_train = tag_tokenizer.texts_to_sequences(train_tags)

x_test = entity_tokenizer.texts_to_sequences(test_sentences)
y_test = tag_tokenizer.texts_to_sequences(test_tags)

x_train_padded = pad_sequences(x_train, maxlen=MAX_LENGTH, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=MAX_LENGTH, padding='post', truncating='post')

y_train_padded = pad_sequences(y_train, maxlen=MAX_LENGTH, padding='post', truncating='post')
y_test_padded = pad_sequences(y_test, maxlen=MAX_LENGTH, padding='post', truncating='post')

# 모델
model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dense(tag_size, activation='softmax')
])



print(model.summary())

try:
    model = keras.models.load_model('models/ner_recognition')
except:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    history = model.fit(x_train_padded, y_train_padded, 
                    batch_size=128, epochs=10, validation_data=(x_test_padded, y_test_padded)
                    )
    model.save('models/ner_recognition')

test_sample = ["EU gave German call to take British people", "Peoples love white christmas"]

test_sample_tokenized = entity_tokenizer.texts_to_sequences(test_sample)
test_sample_padded = pad_sequences(test_sample_tokenized, maxlen=MAX_LENGTH, padding='post', truncating='post')

y_pred = model.predict(test_sample_padded)
y_pred = y_pred.argmax(axis=-1)
index2word = entity_tokenizer.index_word
index2tag = tag_tokenizer.index_word

for i in range(len(test_sample_tokenized)):
    for word, tag in zip([index2word.get(x, '?') for x in test_sample_tokenized[i]], [index2tag.get(x, '?') for x in y_pred[i]]):
        print(f"{word} : {tag}")
