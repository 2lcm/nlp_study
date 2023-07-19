import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical, plot_model
from keras.optimizers import RMSprop

BATCH_SIZE = 64
NUM_SAMPLES = 10000
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
LATENT_DIM = 512

eng_texts = []
kor_inputs = []
kor_targets = []

with open("eng-kor_translation.tsv", 'r') as f:
    lines = f.read().strip().split("\n")
for line in lines:
    _, eng, _, kor = line.split("\t")
    input_kor = '<sos> ' + kor
    target_kor = kor + ' <eos>'

    eng_texts.append(eng)
    kor_inputs.append(input_kor)
    kor_targets.append(target_kor)

tokenizer_eng = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer_eng.fit_on_texts(eng_texts)

eng_sequences = tokenizer_eng.texts_to_sequences(eng_texts)

word2idx_eng = tokenizer_eng.word_index
num_words_eng = min(MAX_VOCAB_SIZE, len(word2idx_eng) + 1)
max_len_eng = max(len(s) for s in eng_sequences)

tokenizer_kor = Tokenizer(num_words=MAX_VOCAB_SIZE, filters="")
tokenizer_kor.fit_on_texts(kor_inputs + kor_targets)

kor_input_sequnces = tokenizer_kor.texts_to_sequences(kor_inputs)
kor_target_sequnces = tokenizer_kor.texts_to_sequences(kor_targets)

word2idx_kor = tokenizer_kor.word_index
num_words_kor = min(MAX_VOCAB_SIZE, len(word2idx_kor) + 1)
max_len_kor = max(len(s) for s in kor_target_sequnces)

encoder_inputs = pad_sequences(eng_sequences, maxlen=max_len_eng, padding='pre')
decoder_inputs = pad_sequences(kor_input_sequnces, maxlen=max_len_kor, padding='post')
decoder_targets = pad_sequences(kor_target_sequnces, maxlen=max_len_kor, padding='post')

embedding_layer_eng = Embedding(num_words_eng, EMBEDDING_DIM)
embedding_layer_kor = Embedding(num_words_kor, EMBEDDING_DIM)

encoder_inputs_ = Input(shape=(max_len_eng), name="Encoder_Input")

x = embedding_layer_eng(encoder_inputs_)
encoder_outputs, h, c = LSTM(LATENT_DIM, return_state=True)(x)
encoder_states = [h, c]
encoder_model = Model(inputs=encoder_inputs_, outputs=encoder_states)
# print(encoder_model.summary())

decoder_inputs_ = Input(shape=(max_len_kor,), name="Decoder_Input")
decoder_inputs_x = embedding_layer_kor(decoder_inputs_)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_kor, activation='softmax', name="Decoder_Output")
decoder_outputs = decoder_dense(decoder_outputs)

model_teacher_forcing = Model(inputs=[encoder_inputs_, decoder_inputs_], outputs=decoder_outputs)
model_teacher_forcing.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(0.001), metrics=['accuracy'])
# print(model_teacher_forcing.summary())

plot_model(model_teacher_forcing, to_file='model.png')

history = model_teacher_forcing.fit([encoder_inputs, decoder_inputs], decoder_targets, batch_size=BATCH_SIZE, epochs=30, validation_split=0.2)

ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], label="Loss")
ax1.plot(history.history['val_loss'], label="Val_Loss")
ax1.legend()

ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], label="Accuracy")
ax2.plot(history.history['val_accuracy'], label="Val_Accuracy")
ax2.legend()

plt.savefig('history.png')