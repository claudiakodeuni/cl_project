import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing.tokenizer import tokenize_dataframe
from preprocessing.pos_tagger import pos_tag_dataframe
from features.feature_extraction import build_tfidf_features
from models.spliting import load_and_split_data


def train_syntactic_model(df_train, df_test):
    """
    Train syntactic SVM model using POS tag unigrams and bigrams.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_test (pd.DataFrame): Test data
    
    Returns:
        (model, vectorizer, X_test, y_test)
    """
    print("=" * 50)
    print("TRAINING SYNTACTIC MODEL (POS Tags)")
    print("=" * 50)
    
    # Tokenize training and test data
    df_train = tokenize_dataframe(df_train, text_column="line")
    df_test = tokenize_dataframe(df_test, text_column="line")
    
    # POS tag training and test data
    print("POS tagging training data...")
    df_train = pos_tag_dataframe(df_train, tokens_column="tokens")
    print("POS tagging test data...")
    df_test = pos_tag_dataframe(df_test, tokens_column="tokens")
    
    # Build TF-IDF features (unigrams + bigrams of POS tags)
    X_train, vectorizer = build_tfidf_features(
        df_train,
        column="pos_sequence",
        ngram_range=(1, 2),  # unigrams and bigrams
        max_features=5000
    )
    y_train = df_train["label"].values
    
    # Transform test set
    X_test = vectorizer.transform(
        df_test["pos_sequence"].apply(lambda x: " ".join(x))
    )
    y_test = df_test["label"].values
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build and train SVM pipeline
    model = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    
    print("Training SVM classifier...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    
    return model, vectorizer, X_test, y_test


if __name__ == "__main__":
    df_train, df_test = load_and_split_data("data/clean/cleaned_data.csv")
    model, vectorizer, X_test, y_test = train_syntactic_model(df_train, df_test)
    
    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/svm_syntactic.joblib")
    joblib.dump(vectorizer, "models/vectorizer_syntactic.joblib")
    print("✓ Model saved to models/svm_syntactic.joblib")
    print("✓ Vectorizer saved to models/vectorizer_syntactic.joblib")
