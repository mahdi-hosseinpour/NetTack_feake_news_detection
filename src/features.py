# src/features.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from src.config import Config


def extract_tfidf_features(df):

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=Config.TFIDF_NGRAM,
        max_features=Config.MAX_TFIDF_FEATURES
    )

    tfidf_matrix = vectorizer.fit_transform(df['statement'])
    tfidf_dense = tfidf_matrix.toarray()

    feature_names = vectorizer.get_feature_names_out()
    print(f" nums of features TF-IDF: {len(feature_names)}")

    return tfidf_dense, feature_names, vectorizer


def reduce_dimensions(features_sparse_or_dense, n_components=Config.PCA_COMPONENTS):

    pca = PCA(n_components=n_components, random_state=Config.SEED)
    reduced = pca.fit_transform(features_sparse_or_dense)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"variance with: {n_components} include: {explained_variance:.4f}")


    return reduced, pca