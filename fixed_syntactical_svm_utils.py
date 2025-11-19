import os
from typing import List
import pandas as pd
import spacy
import re


def get_nlp():
    return spacy.load("es_core_news_sm")


def extract_speaker_lines(path: str, speaker_tag: str = "I:") -> List[str]:
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    return [m.strip() for m in re.findall(rf"^{speaker_tag}\s*(.*)", txt, re.M)]


def process_text_for_syntax(text: str, nlp_model):
    if not isinstance(text, str) or not nlp_model:
        return "", []
    doc = nlp_model(text)
    pos_seq = [
        t.pos_ for t in doc
        if not t.is_punct and not t.is_space and t.pos_ != "INTJ"
    ]
    return " ".join(pos_seq), pos_seq


def build_syntax_dataframe(metadata_path: str, transcripts_dir: str, output_path: str | None = None) -> pd.DataFrame:
    meta = pd.read_excel(metadata_path).set_index("Interview ID")
    rows = []
    nlp_model = get_nlp()

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
            pos_seq_str, pos_seq = process_text_for_syntax(raw, nlp_model)
            if not pos_seq_str:
                continue
            rows.append({
                "Interview ID": iid,
                "Turn": turn,
                "Original Line": raw,
                "POS Sequence": pos_seq_str,
                "City": meta_row.get("City"),
                "Variety": meta_row.get("Variety"),
                "Sex": meta_row.get("Sex (M/F)"),
                "Age group": meta_row.get("Age Group"),
                "Cleaned Line": raw,
            })

    df = pd.DataFrame(rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return df


def prepare_data_for_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Cleaned Line"]).copy()
    df["stratify_label"] = (
        df["Variety"].astype(str) + "_" +
        df["Sex"].astype(str) + "_" +
        df["Age group"].astype(str)
    )
    return df
