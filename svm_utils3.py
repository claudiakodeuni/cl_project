import re
import os
from typing import List

import pandas as pd
try:
    import spacy
except Exception:
    spacy = None


def get_nlp():
    """Load and return spaCy model. Returns None if model unavailable."""
    if not spacy:
        print("spaCy not available. Please install spacy using: pip install spacy")
        return None
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        print("spaCy model 'es_core_news_sm' not found. Run: python -m spacy download es_core_news_sm")
        return None
    except Exception as e:
        print(f"Error loading spaCy model: {str(e)}")
        return None

# Dialect Dictionary
DIALECT_DICT = {"ñawi": "QUECHUA_TOKEN", "wawa": "QUECHUA_TOKEN"}

# explicit interjection set (from user list)
INTERJECTION_SET = {
    "ay","bah","hum","huy","ah","eh","oh","uh",
    "puaf","puaff","puf","puff","uf","uff","aj","puaj",
    "ejem","epa","hala","olé","ajá","ajajá","pche","pchs",
    "pst","buah","am","uhum","uhm","ehm","eeh","emm","aah"
}

# small regex for repeated-letter fillers like 'uhhh' or 'eeh'
INTERJECTION_RE = re.compile(r"\b([aeiouh])\1{1,}\b", re.I)


def clean_line(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"/", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def remove_interjections(s: str) -> str:
    if not s:
        return s
    toks = s.split()
    kept = [t for t in toks if not (t.lower() in INTERJECTION_SET or INTERJECTION_RE.fullmatch(t))]
    return " ".join(kept)


def standardize_tokens(s: str) -> str:
    return " ".join(DIALECT_DICT.get(t, t) for t in s.split())


def _filter_tokens(toks):
    """Keep only alphabetic tokens with length > 1 and normalize to lower-case."""
    out = []
    if not toks:
        return out
    # if saved as a string, split; otherwise assume iterable of strings
    if isinstance(toks, str):
        toks = toks.split()
    for t in toks:
        if not isinstance(t, str):
            continue
        tt = t.lower()
        if tt.isalpha() and len(tt) > 1:
            out.append(tt)
    return out


def extract_speaker_lines(path: str, speaker_tag: str = "I:") -> List[str]:
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    return [m.strip() for m in re.findall(rf"^{speaker_tag}\s*(.*)", txt, re.M)]


def process_text(raw: str, nlp_model=None, use_pos: bool = False):
    """Process a single text line with optional NLP model usage.
    
    Args:
        raw: Raw text input
        nlp_model: Optional spaCy model for NER and POS tagging
        use_pos: Whether to extract POS tags
    
    Returns:
        tuple: (cleaned_text, tokens, pos_tags)
    """
    # Run NER on the raw text before removing punctuation/lowercasing
    pos = []
    text = raw if isinstance(raw, str) else ""

    if nlp_model:
        try:
            doc = nlp_model(text)  # run NER on original text (preserve punctuation/case)
            # Replace locations with marker using token-level entity types
            text = " ".join("location" if tok.ent_type_ in ("LOC", "GPE") else tok.text for tok in doc)
            if use_pos:
                pos = [(t.text, t.pos_) for t in doc]
        except Exception as e:
            # If spaCy fails on this line, continue with cleaning but warn once
            print(f"Warning: spaCy NER failed for line: {str(e)}")

    # Now apply the cleaning pipeline (remove punctuation, lower-case, etc.)
    text = clean_line(text)
    text = remove_interjections(text)
    text = standardize_tokens(text)
    tokens = _filter_tokens(text.split())

    return text, tokens, pos

def build_cleaned_dataframe(metadata_path: str, transcripts_dir: str, output_path: str | None = None, use_spacy: bool = True, use_pos: bool = False) -> pd.DataFrame:
    """Build DataFrame with cleaned lines and token lists.

    - use_spacy: if True, attempt to run spaCy NER to replace LOC/GPE with 'location'.
    - use_pos: if True, collect (token, pos) pairs; turning it off speeds processing.
    """
    meta = pd.read_excel(metadata_path).set_index("Interview ID")
    rows = []
    nlp_model = get_nlp() if use_spacy else None
    
    # If using spaCy, prepare the model and warn if not available
    if use_spacy and nlp_model is None:
        print("\nWARNING: SpaCy model not available - NER processing will be skipped!\n")
    elif nlp_model:
        disable = [c for c in nlp_model.pipe_names if c != "ner"]
        nlp_model.disable_pipes(*disable)  # More efficient than passing disable list each time

    for fn in sorted(os.listdir(transcripts_dir)):
        if not fn.lower().endswith('.txt'):
            continue
        iid = fn[:-4]
        if iid not in meta.index:
            continue
        filepath = os.path.join(transcripts_dir, fn)
        raw_lines = extract_speaker_lines(filepath)
        if not raw_lines:
            continue

        meta_row = meta.loc[iid]
        for turn, raw in enumerate(raw_lines, 1):
            text, tokens, pos = process_text(raw, nlp_model, use_pos)
            rows.append({
                "Interview ID": iid,
                "Turn": turn,
                "Original Line": raw,
                "Cleaned Line": text,
                "Tokens": tokens,
                "POS Tags": pos,
                "City": meta_row.get("City"),
                "Variety": meta_row.get("Variety"),
                "Sex": meta_row.get("Sex (M/F)"),
                "Age group": meta_row.get("Age Group"),
            })

    df = pd.DataFrame(rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df


def prepare_data_for_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Cleaned Line"]).copy()
    df["stratify_label"] = df["Variety"].astype(str) + "_" + df["Sex"].astype(str) + "_" + df["Age group"].astype(str)
    return df


__all__ = ["clean_line","remove_interjections","build_cleaned_dataframe","prepare_data_for_split"]
