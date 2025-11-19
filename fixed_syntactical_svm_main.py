import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
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
    X = [" ".join(seq) if isinstance(seq, list) else seq for seq in X]
    y = df["Variety"].values
    groups = df["Interview ID"].values

    print("\n5-fold GroupKFold CV (syntaktische Features only):")
    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = [X[i] for i in tr_idx], [X[i] for i in te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=10000
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        fold_scores.append((acc, f1_micro, f1_macro))

        print(f"\nFold {fold}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Micro:  {f1_micro:.4f}")
        print(f"  F1 Macro:  {f1_macro:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    arr = np.array(fold_scores)
    print("\nOverall Performance (mean ± std):")
    print(f"Accuracy:  {arr[:,0].mean():.4f} (±{arr[:,0].std():.4f})")
    print(f"F1 Micro:  {arr[:,1].mean():.4f} (±{arr[:,1].std():.4f})")
    print(f"F1 Macro:  {arr[:,2].mean():.4f} (±{arr[:,2].std():.4f})")


if __name__ == "__main__":
    main()
