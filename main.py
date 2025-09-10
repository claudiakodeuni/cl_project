import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from utils import build_cleaned_dataframe, prepare_data_for_split

# --- Paths ---
metadata_path = "data/metadata.xlsx"
transcripts_dir = "data/raw_transcripts"
output_dir = "data/cleaned_csv"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "cleaned_transcripts.csv")

# --- Preprocessing ---
df = build_cleaned_dataframe(metadata_path, transcripts_dir, output_csv)
df = prepare_data_for_split(df)

# --- Train/Test Split ---
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['stratify_label'],
    random_state=42
)

X_train_text = train_df['Line']
y_train = train_df['Variety']
X_test_text = test_df['Line']
y_test = test_df['Variety']

# --- Model Selection and Hyperparameters ---
# Here we choose Multinomial Naive Bayes with default parameters
# You could later tune alpha or other hyperparameters if needed
vectorizer = CountVectorizer(ngram_range=(1)) # Unigrams
model = MultinomialNB(alpha=1.0) # Laplace smoothing

# --- Training ---
X_train_vect = vectorizer.fit_transform(X_train_text)
model.fit(X_train_vect, y_train)

# --- Evaluation on Test Set ---
X_test_vect = vectorizer.transform(X_test_text)
y_pred_test = model.predict(X_test_vect)

test_accuracy = accuracy_score(y_test, y_pred_test)
f1_micro_test = f1_score(y_test, y_pred_test, average='micro')
f1_macro_test = f1_score(y_test, y_pred_test, average='macro')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 (Micro): {f1_micro_test:.4f}")
print(f"Test F1 (Macro): {f1_macro_test:.4f}")

# --- Stratified K-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(df, df['stratify_label'])):
    fold_train = df.iloc[train_index].dropna(subset=['Line']).copy()
    fold_val = df.iloc[val_index].dropna(subset=['Line']).copy()

    X_fold_train_vect = vectorizer.fit_transform(fold_train['Line'])
    y_fold_train = fold_train['Variety']

    X_fold_val_vect = vectorizer.transform(fold_val['Line'])
    y_fold_val = fold_val['Variety']

    model_fold = MultinomialNB()
    model_fold.fit(X_fold_train_vect, y_fold_train)

    y_pred_val = model_fold.predict(X_fold_val_vect)

    val_accuracy = accuracy_score(y_fold_val, y_pred_val)
    val_f1_micro = f1_score(y_fold_val, y_pred_val, average='micro')
    val_f1_macro = f1_score(y_fold_val, y_pred_val, average='macro')

    print(f"Fold {fold+1} Accuracy: {val_accuracy:.4f}, F1 Micro: {val_f1_micro:.4f}, F1 Macro: {val_f1_macro:.4f}")