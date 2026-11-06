import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_chunks(sentences, chunk_size=5):

    chunks = []

    for i in range(0, len(sentences), chunk_size):

        chunk = sentences[i:i + chunk_size]

        text = " ".join(
            [" ".join(sentence) for sentence in chunk]
        )

        if text.strip():
            chunks.append(text)

    return chunks


def train_word2vec(sentences):

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=30,
        seed=42
    )

    return model


def calculate_word2vec_similarity(model, search_word, top_n=10):

    if search_word not in model.wv:
        return pd.DataFrame(
            columns=['Word', 'Word2Vec Score']
        )

    results = model.wv.most_similar(
        search_word,
        topn=top_n
    )

    return pd.DataFrame(
        results,
        columns=['Word', 'Word2Vec Score']
    )


def calculate_tfidf_similarity(chunks, search_word, top_n=10):

    vectorizer = TfidfVectorizer(
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
    )

    matrix = vectorizer.fit_transform(chunks)

    vocabulary = vectorizer.get_feature_names_out()

    if search_word not in vectorizer.vocabulary_:
        return pd.DataFrame(
            columns=['Word', 'TFIDF Score']
        )

    word_index = vectorizer.vocabulary_[search_word]

    
    search_vector = matrix[:, word_index]

    
    scores = cosine_similarity(
        matrix.T,
        search_vector.T
    ).flatten()

    results = []

    for index, score in enumerate(scores):

        word = vocabulary[index]

        if word != search_word and score > 0:
            results.append((word, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return pd.DataFrame(
        results[:top_n],
        columns=['Word', 'TFIDF Score']
    )


def find_similar_words(
    sentences,
    chunks,
    search_word,
    top_n=10
):

    search_word = search_word.lower().strip()


    model = train_word2vec(sentences)

    
    w2v_results = calculate_word2vec_similarity(
        model,
        search_word,
        top_n=top_n
    )

    
    tfidf_results = calculate_tfidf_similarity(
        chunks,
        search_word,
        top_n=top_n
    )

    
    combined = pd.merge(
        w2v_results,
        tfidf_results,
        on='Word',
        how='outer'
    )

    if combined.empty:
        return combined, model

    combined = combined.fillna(0)

    
    combined['Combined Score'] = (
        0.6 * combined['Word2Vec Score']
        + 0.4 * combined['TFIDF Score']
    )

    combined = combined.sort_values(
        by='Combined Score',
        ascending=False
    )

    combined = combined.head(top_n)

    return combined.reset_index(drop=True), model