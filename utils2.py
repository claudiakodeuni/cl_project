import re
import os
import pandas as pd
import spacy

# Load spaCy model (e.g., Spanish)
# You can use 'es_core_news_sm' for Spanish; install it first with:
# python -m spacy download es_core_news_sm
nlp = spacy.load("es_core_news_sm")

# Example dialect-specific dictionary (extend as needed)
DIALECT_DICT = {
    "ñawi": "QUECHUA_TOKEN",
    "wawa": "QUECHUA_TOKEN",
    "gringasho": "LOANWORD_TOKEN"
}

# Stopwords (optional): use spaCy's or define custom
STOPWORDS = nlp.Defaults.stop_words

# -------------------------------------------------------
# 1. Clean and normalize text lines
# -------------------------------------------------------
def clean_line(line):
    line = re.sub(r"<[^>]+>", "", line)  # Remove annotation tags
    line = re.sub(r"/", "", line)        # Remove slashes
    line = re.sub(r"[^\w\s]", "", line)  # Remove punctuation
    line = re.sub(r"\s+", " ", line)     # Normalize whitespace
    line = line.strip().lower()          # Lowercase
    return line

# -------------------------------------------------------
# 2. Named Entity Recognition (NER) + Named Entity Tagging
# -------------------------------------------------------
def apply_ner(text):
    """Identify named entities and replace locations / proper names with tags."""
    doc = nlp(text)
    new_tokens = []
    for ent in doc.ents:
        # Example: replace any location name with a LOCATION tag
        if ent.label_ in ["LOC", "GPE"]:
            text = text.replace(ent.text.lower(), "location")
        # You could add other entity standardizations here (e.g., PERSON → "person")
    return text

# -------------------------------------------------------
# 3. Dialect-Specific Token Standardization
# -------------------------------------------------------
def standardize_dialect_tokens(text):
    tokens = text.split()
    standardized = [DIALECT_DICT.get(tok, tok) for tok in tokens]
    return " ".join(standardized)

# -------------------------------------------------------
# 4. Tokenization and optional stopword handling
# -------------------------------------------------------
def tokenize_text(text, remove_stopwords=False):
    doc = nlp(text)
    if remove_stopwords:
        tokens = [t.text for t in doc if not t.is_stop]
    else:
        tokens = [t.text for t in doc]
    return tokens

# -------------------------------------------------------
# 5. POS Tagging
# -------------------------------------------------------
def pos_tag_text(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# -------------------------------------------------------
# Metadata loading
# -------------------------------------------------------
def load_metadata(metadata_path):
    metadata = pd.read_excel(metadata_path)
    return metadata.set_index("Interview ID")

# -------------------------------------------------------
# Extract lines spoken by interviewee only ("I:")
# -------------------------------------------------------
def extract_speaker_lines(filepath, speaker_tag="I:"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    lines = re.findall(rf"^{speaker_tag}\s*(.*)", content, re.MULTILINE)
    cleaned = [clean_line(line) for line in lines]
    return cleaned

# -------------------------------------------------------
# Gather transcript rows with preprocessing pipeline
# -------------------------------------------------------
def gather_transcript_rows(transcripts_dir, metadata, remove_stopwords=False):
    rows = []
    for filename in os.listdir(transcripts_dir):
        if not filename.endswith(".txt"):
            continue

        interview_id = filename.replace(".txt", "")
        filepath = os.path.join(transcripts_dir, filename)

        if interview_id not in metadata.index:
            print(f"Entrevista {interview_id} no está en los metadatos. Se ignora.")
            continue

        meta = metadata.loc[interview_id]
        cleaned_lines = extract_speaker_lines(filepath)

        for turn, line in enumerate(cleaned_lines, 1):
            # NER standardization
            line_ner = apply_ner(line)

            # Dialectal token normalization
            line_standardized = standardize_dialect_tokens(line_ner)

            # Tokenization (with or without stopword removal)
            tokens = tokenize_text(line_standardized, remove_stopwords=remove_stopwords)

            # POS tagging
            pos_tags = pos_tag_text(line_standardized)

            rows.append({
                "Interview ID": interview_id,
                "Turn": turn,
                "Original Line": line,
                "Cleaned Line": line_standardized,
                "Tokens": tokens,
                "POS Tags": pos_tags,
                "City": meta["City"],
                "Variety": meta["Variety"],
                "Sex": meta["Sex (M/F)"],
                "Age group": meta["Age Group"],
                "Educational level": meta["Educational level"]
            })
    return rows

# -------------------------------------------------------
# Build DataFrame and save to CSV if needed
# -------------------------------------------------------
def build_cleaned_dataframe(metadata_path, transcripts_dir, output_path=None, remove_stopwords=False):
    metadata = load_metadata(metadata_path)
    rows = gather_transcript_rows(transcripts_dir, metadata, remove_stopwords=remove_stopwords)
    df = pd.DataFrame(rows)
    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"CSV file created successfully: {output_path}")
    return df

# -------------------------------------------------------
# Prepare data for stratified splitting
# -------------------------------------------------------
def prepare_data_for_split(df):
    df = df.dropna(subset=['Cleaned Line']).copy()
    df['stratify_label'] = (
        df['Variety'].astype(str) + "_" +
        df['Sex'].astype(str) + "_" +
        df['Age group'].astype(str)
    )
    return df
