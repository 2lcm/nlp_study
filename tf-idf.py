import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    'I love my dog',
    'I love my cat',
    'I love my dog and love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_sentences = tfidf_vectorizer.fit_transform(sentences)
tfidf_vect_sentences = tfidf_sentences.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()

df = pd.DataFrame(tfidf_vect_sentences, columns=feature_names)
print(df)