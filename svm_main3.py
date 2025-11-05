import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from svm_utils2 import build_cleaned_dataframe, prepare_data_for_split


def main():
    base = Path(__file__).resolve().parent
    metadata_path = base / "data" / "metadata_cl.xlsx"
    transcripts_dir = base / "data" / "raw_data"
    output_csv = base / "data" / "clean_data" / "cleaned_transcripts.csv"

    os.makedirs(output_csv.parent, exist_ok=True)
    df = build_cleaned_dataframe(str(metadata_path), str(transcripts_dir), output_path=str(output_csv), use_spacy=True)
    df = prepare_data_for_split(df)

    # Convert tokens to space-separated text for TfidfVectorizer
    # The vectorizer's token_pattern will handle filtering
    df["VectorText"] = df["Tokens"].apply(lambda t: " ".join(t if isinstance(t, list) else str(t).split()))
    X = df["VectorText"].values
    y = df["Variety"].values

    # 5-fold Cross-Validation
    print("\n5-fold Cross-Validation Results:")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        # Split data for this fold
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(
            analyzer="word", 
            ngram_range=(1, 1), 
            token_pattern=r"(?u)\b\w{2,}\b", 
            max_features=15000
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train and evaluate model
        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        scores = (
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average="micro"),
            f1_score(y_test, y_pred, average="macro")
        )
        fold_scores.append(scores)
        
        print(f"Fold {fold}:")
        print(f"  Accuracy:  {scores[0]:.4f}")
        print(f"  F1 Micro:  {scores[1]:.4f}")
        print(f"  F1 Macro:  {scores[2]:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Calculate and display mean scores with standard deviation
    fold_arr = np.array(fold_scores)
    print("\nOverall Performance (mean ± std):")
    print(f"Accuracy:  {fold_arr[:,0].mean():.4f} (±{fold_arr[:,0].std():.4f})")
    print(f"F1 Micro:  {fold_arr[:,1].mean():.4f} (±{fold_arr[:,1].std():.4f})")
    print(f"F1 Macro:  {fold_arr[:,2].mean():.4f} (±{fold_arr[:,2].std():.4f})")

    # Train final model on full dataset for feature analysis
    print("\nTraining final model on full dataset for feature analysis...")
    final_vectorizer = TfidfVectorizer(
        analyzer="word", 
        ngram_range=(1, 1), 
        token_pattern=r"(?u)\b\w{2,}\b", 
        max_features=15000
    )
    X_tfidf = final_vectorizer.fit_transform(X)
    final_model = LinearSVC(random_state=42, max_iter=10000)
    final_model.fit(X_tfidf, y)

    # Analyze top features per class
    print("\nTop features per class:")
    feat_names = np.array(final_vectorizer.get_feature_names_out())
    if hasattr(final_model, "coef_"):
        coefs = final_model.coef_
        if coefs.shape[0] == 1 and len(final_model.classes_) == 2:
            # Binary classification case
            row = coefs[0]
            top_pos = np.argsort(row)[-10:][::-1]
            top_neg = np.argsort(row)[:10]
            print(f"{final_model.classes_[1]}: {', '.join(feat_names[top_pos])}")
            print(f"{final_model.classes_[0]}: {', '.join(feat_names[top_neg])}")
        else:
            # Multi-class case
            for i, cls in enumerate(final_model.classes_):
                row = coefs[i]
                top_idx = np.argsort(row)[-10:][::-1]
                print(f"{cls}: {', '.join(feat_names[top_idx])}")


if __name__ == "__main__":
    main()
