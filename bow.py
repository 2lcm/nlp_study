import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    'I love my dog',
    'I love my cat',
    'I love my dog and love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

count_ventorizer = CountVectorizer()
features = count_ventorizer.fit_transform(sentences)
vectorized_sentences = features.toarray()
feature_names = count_ventorizer.get_feature_names_out()

df = pd.DataFrame(vectorized_sentences, columns=feature_names)
print(df)