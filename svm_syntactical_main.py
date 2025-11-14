import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from svm_syntactical_utils import build_syntax_dataframe, prepare_data_for_split

def main():
    base = Path(__file__).resolve().parent
    metadata_path = base / "data" / "metadata_cl.xlsx"
    transcripts_dir = base / "data" / "raw_data"
    output_csv = base / "data" / "clean_data" / "syntax_only.csv"

    os.makedirs(output_csv.parent, exist_ok=True)

    df = build_syntax_dataframe(
        str(metadata_path),
        str(transcripts_dir),
        output_path=str(output_csv)
    )
    df = prepare_data_for_split(df)

    X = df["POS Sequence"].values
    y = df["Variety"].values

    print("\n5-fold Cross-Validation (syntactic features only):")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),   
            token_pattern=r"(?u)\b[a-zA-Z]+\b",
            max_features=10000
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train SVM
        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        fold_scores.append((acc, f1_micro, f1_macro))

        print(f"Fold {fold}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Micro:  {f1_micro:.4f}")
        print(f"  F1 Macro:  {f1_macro:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Summary
    arr = np.array(fold_scores)
    print("\nOverall Performance (mean ± std):")
    print(f"Accuracy:  {arr[:,0].mean():.4f} (±{arr[:,0].std():.4f})")
    print(f"F1 Micro:  {arr[:,1].mean():.4f} (±{arr[:,1].std():.4f})")
    print(f"F1 Macro:  {arr[:,2].mean():.4f} (±{arr[:,2].std():.4f})")

    # Optional: top POS n-grams per dialect
    print("\nTop syntactic patterns per class:")
    vectorizer_full = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        max_features=10000
    )
    X_full = vectorizer_full.fit_transform(X)
    model_full = LinearSVC(random_state=42, max_iter=10000)
    model_full.fit(X_full, y)

    feat_names = np.array(vectorizer_full.get_feature_names_out())

    if hasattr(model_full, "coef_"):
        coefs = model_full.coef_
        classes = model_full.classes_

    # Case 1: binary classification
    if coefs.shape[0] == 1 and len(classes) == 2:
        row = coefs[0]
        top_pos = np.argsort(row)[-10:][::-1]
        top_neg = np.argsort(row)[:10]
        print(f"\nTop syntactic patterns per class:")
        print(f"{classes[1]}: {', '.join(feat_names[top_pos])}")
        print(f"{classes[0]}: {', '.join(feat_names[top_neg])}")

    # Fall 2: more then two classes 
    else:
        print(f"\nTop syntactic patterns per class:")
        for i, cls in enumerate(classes):
            row = coefs[i]
            top_idx = np.argsort(row)[-10:][::-1]
            print(f"{cls}: {', '.join(feat_names[top_idx])}")


if __name__ == "__main__":
    main()
