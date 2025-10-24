import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# === Datapreperation (based on input of utils2) ===
from utils2 import build_cleaned_dataframe, prepare_data_for_split

# --- paths ---
metadata_path = "C:/Users/Asus/Documents/Uni_Trento/03_Semester/Computational_Linguistics/svm_2/data/metadata_cl.xlsx"
transcripts_dir = "C:/Users/Asus/Documents/Uni_Trento/03_Semester/Computational_Linguistics/svm_2/data/raw_data"
output_csv = "C:/Users/Asus/Documents/Uni_Trento/03_Semester/Computational_Linguistics/svm_2/data/clean_data/cleaned_transcripts.csv"

# --- loading and cleaning of the data
df = build_cleaned_dataframe(metadata_path, transcripts_dir, output_path=output_csv, remove_stopwords=True)
df = prepare_data_for_split(df)

# --- features and labels
X = df["Cleaned Line"].values
y = df["Variety"].values

# --- train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df["stratify_label"]
)

# --- TF-IDF Vektorisierung ---
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---SVM Training ---
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# --- Test Evaluation ---
y_pred = svm_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1_micro = f1_score(y_test, y_pred, average="micro")
test_f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n=== Test Results ===")
print(f"Accuracy:     {test_accuracy:.4f}")
print(f"F1 (Micro):   {test_f1_micro:.4f}")
print(f"F1 (Macro):   {test_f1_macro:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 5-Fold Cross Validation ---
print("\n=== 5-Fold Cross Validation ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    X_train_fold_tfidf = vectorizer.fit_transform(X_train_fold)
    X_test_fold_tfidf = vectorizer.transform(X_test_fold)

    model = LinearSVC(random_state=42)
    model.fit(X_train_fold_tfidf, y_train_fold)
    y_pred_fold = model.predict(X_test_fold_tfidf)

    acc = accuracy_score(y_test_fold, y_pred_fold)
    f1_micro = f1_score(y_test_fold, y_pred_fold, average="micro")
    f1_macro = f1_score(y_test_fold, y_pred_fold, average="macro")

    fold_results.append((acc, f1_micro, f1_macro))
    print(f"Fold {fold}: Accuracy={acc:.4f}, F1_micro={f1_micro:.4f}, F1_macro={f1_macro:.4f}")

fold_df = pd.DataFrame(fold_results, columns=["Accuracy", "F1_Micro", "F1_Macro"])
print("\nMean over 5 folds:")
print(fold_df.mean())

# --- Schritt 8: Top 10 Features pro Klasse ---
print("\n=== Top 10 Features per Class ===")
feature_names = np.array(vectorizer.get_feature_names_out())
coef = svm_model.coef_[0]
classes = svm_model.classes_
print("SVM Klassen:", classes.tolist())

# Klasse 0 (negative Gewichte)
top10_class0_idx = np.argsort(coef)[:10]
top10_class0 = pd.DataFrame({
    "Feature": feature_names[top10_class0_idx],
    "Weight": coef[top10_class0_idx]
}).sort_values(by="Weight", ascending=True)

# Klasse 1 (positive Gewichte)
top10_class1_idx = np.argsort(coef)[-10:]
top10_class1 = pd.DataFrame({
    "Feature": feature_names[top10_class1_idx],
    "Weight": coef[top10_class1_idx]
}).sort_values(by="Weight", ascending=False)

print(f"\nTop 10 Features für '{classes[0]}' (negativ gewichtete Merkmale):")
print(top10_class0.to_string(index=False))

print(f"\nTop 10 Features für '{classes[1]}' (positiv gewichtete Merkmale):")
print(top10_class1.to_string(index=False))
