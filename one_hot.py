import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

sentences = [
    'I love my dog',
    'I love my cat',
    'I love my dog and love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=5, padding='post', truncating='post')
print(padded)

for sequence in sequences:
    sent = []
    for idx in sequence:
        sent.append(tokenizer.index_word[idx])
    print(' '.join(sent))

one_hot = to_categorical(padded)
print(one_hot)