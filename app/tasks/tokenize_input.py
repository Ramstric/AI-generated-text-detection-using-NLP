import numpy as np

import gensim

from nltk.tokenize import word_tokenize

w2vModel = gensim.models.Word2Vec.load("models/word2vec/word2vec.model")

def tokenize(text: str):
    tokens = word_tokenize(text.lower())
    text_vector = vectorize_text(tokens, w2vModel, w2vModel.vector_size)
    text_vector = np.array(text_vector).reshape(1, -1)
    return text_vector

def vectorize_text(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)