from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt
import pandas as pd
import sentencepiece as spm

sentences_E = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'I was born in Korea and graduated university in USA'
]

sentences_K = [
    '코로나가 심하다',
    '코비드-19가 심하다',
    '아버지가방에들어가신다',
    '아버지가 방에 들어가신다',
    '너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다.'
]

tokenizer = Tokenizer(num_words=100, oov_token='OOV')
tokenizer.fit_on_texts(sentences_E)
print(tokenizer.index_word)
tokenizer.fit_on_texts(sentences_K)
print(tokenizer.index_word)

okt = Okt()

temp_X =[]
for sent in sentences_K:
    temp_X.append(okt.morphs(sent))

print(temp_X)


# 네이버 리뷰
DATA_TRAIN_PATH = "data/ratings_train.txt"

train_data = pd.read_csv(DATA_TRAIN_PATH, sep='\t', quoting=3)
train_data.dropna(inplace=True)
print(train_data.shape)
print(train_data.head())