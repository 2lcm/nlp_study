from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GRU
import tensorflow as tf
import numpy as np

tf.keras.utils.disable_interactive_logging()

B=2 # batch size
T=5 # time steps
D=1 # features
U=3 # LSTM units

X = np.random.randn(B, T, D)

def lstm(return_sequences=False, return_state=False):
    inp = Input(shape=(T,D))
    out = LSTM(U, return_sequences=return_sequences, return_state=return_state)(inp)
    model = Model(inputs=inp, outputs=out)
    return model.predict(X)

o = lstm(return_sequences=False, return_state=False)
print(o.shape)
print(o)

o, h, c = lstm(return_sequences=False, return_state=True)
print(o.shape)
print(h.shape)
print(c.shape)

o = lstm(return_sequences=True, return_state=False)
print(o.shape)
print(o)

o, h, c = lstm(return_sequences=True, return_state=True)

print(o.shape)
print(h.shape)
print(c.shape)