from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(max_features=20_000, ngram_range=(1, 2)):
    """
    TF-IDF vectorizer: top max_features, unigrams & bigrams, English stop-words removed.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
