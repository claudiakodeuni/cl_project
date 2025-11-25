import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def prepare_text_for_tfidf(token_list):
    """
    Converts a list of tokens (or POS tags) into a single string
    suitable for TF-IDF vectorization.

    Args:
        token_list (list of str): List of tokens or POS tags.

    Returns:
        str: Tokens joined into a space-separated string.
    """
    if isinstance(token_list, list):
        return " ".join(token_list)
    return token_list  # already a string


def build_tfidf_features(df, column="tokens", ngram_range=(1, 1), max_features=10000):
    """
    Create TF-IDF features from a dataframe column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to vectorize ('tokens' or 'pos_sequence').
        ngram_range (tuple): (min_n, max_n) for TF-IDF ngrams.
        max_features (int): Maximum number of features to keep.

    Returns:
        X_tfidf: TF-IDF matrix (sparse)
        vectorizer: fitted TfidfVectorizer object
    """
    # Convert token lists to strings for TF-IDF
    text_data = df[column].apply(prepare_text_for_tfidf)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features
    )

    X_tfidf = vectorizer.fit_transform(text_data)

    return X_tfidf, vectorizer


def transform_new_text(new_token_lists, vectorizer):
    """
    Transform new token lists (or POS sequences) into TF-IDF using an existing vectorizer.

    Args:
        new_token_lists (list of list of str): New tokens/POS sequences.
        vectorizer (TfidfVectorizer): Already fitted vectorizer.

    Returns:
        TF-IDF matrix (sparse)
    """
    text_data = [" ".join(lst) if isinstance(lst, list) else lst for lst in new_token_lists]
    return vectorizer.transform(text_data)


if __name__ == "__main__":
    # Example usage
    csv_path = "data/clean/cleaned_data.csv"
    df = pd.read_csv(csv_path)

    # Tokenization example
    if "tokens" not in df.columns:
        from preprocessing.tokenizer import tokenize_dataframe
        df = tokenize_dataframe(df)

    # POS tagging example
    if "pos_sequence" not in df.columns:
        from preprocessing.pos_tagger import pos_tag_dataframe
        df = pos_tag_dataframe(df)

    # Lexical TF-IDF
    X_tokens, vectorizer_tokens = build_tfidf_features(df, column="tokens", ngram_range=(1,1))
    print("Lexical TF-IDF shape:", X_tokens.shape)

    # POS TF-IDF (syntactic)
    X_pos, vectorizer_pos = build_tfidf_features(df, column="pos_sequence", ngram_range=(1,2))
    print("POS TF-IDF shape:", X_pos.shape)
