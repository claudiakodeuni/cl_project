import re
import os
from typing import List
import pandas as pd
import spacy

def get_nlp():
        return spacy.load("es_core_news_sm")
    
def extract_speaker_lines(path: str, speaker_tag: str = "I:") -> List[str]:
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    return [m.strip() for m in re.findall(rf"^{speaker_tag}\s*(.*)", txt, re.M)]

INTERJECTION_SET = {
    "ay","bah","hum","huy","ah","eh","oh","uh",
    "puaf","puaff","puf","puff","uf","uff","aj","puaj",
    "ejem","epa","hala","olé","ajá","ajajá","pche","pchs",
    "pst","buah","am","uhum","uhm","ehm","eeh","emm","aah"
}
INTERJECTION_RE = re.compile(r"\b([aeiouh])\1{1,}\b", re.I)

def remove_interjections(s: str) -> str:
    if not s:
        return s
    toks = s.split()
    kept = [t for t in toks if not (t.lower() in INTERJECTION_SET or INTERJECTION_RE.fullmatch(t))]
    return " ".join(kept)

def clean_line(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"/", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _filter_tokens(toks):
    if isinstance(toks, str):
        toks = toks.split()
    return [t.lower() for t in toks if t.isalpha() and len(t) > 1]


def process_text(raw: str, nlp_model=None, use_pos: bool = False):
    text = raw
    if nlp_model:
            doc = nlp_model(text) 
            text = " ".join("location" if tok.ent_type_ in ("LOC", "GPE") else tok.text for tok in doc)
        
        
    text = clean_line(text)
    text = remove_interjections(text)
    tokens = _filter_tokens(text.split())

    return text, tokens


def build_cleaned_dataframe(metadata_path: str, transcripts_dir: str, output_path: str | None = None, use_spacy: bool = True, use_pos: bool = False) -> pd.DataFrame:
    meta = pd.read_excel(metadata_path).set_index("Interview ID")
    rows = []
    nlp_model = get_nlp() if use_spacy else None
    
    if use_spacy and nlp_model is None:
        print("\nWARNING: SpaCy model not available - NER processing will be skipped!\n")
    elif nlp_model:
        disable = [c for c in nlp_model.pipe_names if c != "ner"]
        nlp_model.disable_pipes(*disable)  

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
            text, tokens = process_text(raw, nlp_model=nlp_model, use_pos=use_pos)

            rows.append({
                "Interview ID": iid,
                "Turn": turn,
                "Original Line": raw,
                "Cleaned Line": text,
                "Tokens": tokens,
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
