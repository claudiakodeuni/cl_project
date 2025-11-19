from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from svm_lexical_utils import build_cleaned_dataframe


def main():
    base = Path(__file__).resolve().parent
    metadata_path = base / "data" / "metadata_cl.xlsx"
    transcripts_dir = base / "data" / "raw_data"
    output_csv = base / "data" / "clean_data" / "cleaned_transcripts.csv"

    df = build_cleaned_dataframe(
        str(metadata_path),
        str(transcripts_dir),
        output_path=str(output_csv),
        use_spacy=True
    )

    df["VectorText"] = df["Tokens"].apply(lambda t: " ".join(t))
    X = df["VectorText"].values
    y = df["Variety"].values
    groups = df["Interview ID"].values 

    print("\n5-fold GroupKFold Results:")
    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            token_pattern=r"(?u)\b\w{2,}\b",
            max_features=10000
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        scores = (
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average="micro"),
            f1_score(y_test, y_pred, average="macro")
        )
        fold_scores.append(scores)

        print(f"\nFold {fold}:")
        print(f"  Accuracy:  {scores[0]:.4f}")
        print(f"  F1 Micro:  {scores[1]:.4f}")
        print(f"  F1 Macro:  {scores[2]:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    fold_arr = np.array(fold_scores)
    print("\nOverall Performance (mean ± std):")
    print(f"Accuracy:  {fold_arr[:,0].mean():.4f} (±{fold_arr[:,0].std():.4f})")
    print(f"F1 Micro:  {fold_arr[:,1].mean():.4f} (±{fold_arr[:,1].std():.4f})")
    print(f"F1 Macro:  {fold_arr[:,2].mean():.4f} (±{fold_arr[:,2].std():.4f})")

    print("\nTraining final model on full dataset...")
    final_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w{2,}\b",
        max_features=15000
    )
    X_tfidf = final_vectorizer.fit_transform(X)
    final_model = LinearSVC(random_state=42, max_iter=10000)
    final_model.fit(X_tfidf, y)

    print("\nTop features for binary classifier:")

    feat_names = np.array(final_vectorizer.get_feature_names_out())
    coefs = final_model.coef_[0]

    top_pos = np.argsort(coefs)[-10:][::-1]   
    top_neg = np.argsort(coefs)[:10]           

    cls_pos = final_model.classes_[1]
    cls_neg = final_model.classes_[0]

    print(f"\nClass {cls_pos}: {', '.join(feat_names[top_pos])}")
    print(f"Class {cls_neg}: {', '.join(feat_names[top_neg])}")


if __name__ == "__main__":
    main()
