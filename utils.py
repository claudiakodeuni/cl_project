import re
import os
import pandas as pd

# Function to clean and normalize text lines
def clean_line(line):
    line = re.sub(r"<[^>]+>", "", line)  # Remove annotation tags
    line = re.sub(r"/", "", line)        # Remove slashes
    line = re.sub(r"[^\w\s]", "", line)  # Remove punctuation
    line = line.lower()                  # Lowercase
    line = re.sub(r"\s+", " ", line).strip()  # Normalize whitespace
    return line

# Function to load metadata from an Excel file
def load_metadata(metadata_path):
    metadata = pd.read_excel(metadata_path)
    return metadata.set_index("Interview ID")

# Function to extract lines spoken by a specific speaker
def extract_speaker_lines(filepath, speaker_tag="I:"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    lines = re.findall(rf"^{speaker_tag}\s*(.*)", content, re.MULTILINE)
    cleaned = [clean_line(line) for line in lines]
    return cleaned

# Function to gather all transcript rows with metadata
def gather_transcript_rows(transcripts_dir, metadata):
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
            rows.append({
                "Line": line,
                "Turn": turn,
                "Interview ID": interview_id,
                "City": meta["City"],
                "Variety": meta["Variety (L/Q)"],
                "Sex": meta["Sex (M/F)"],
                "Age group": meta["Age Group"],
                "Educational level": meta["Educational level"]
            })
    return rows

# Composite function to build cleaned DataFrame and optionally save as CSV
def build_cleaned_dataframe(metadata_path, transcripts_dir, output_path=None):
    """Create a cleaned DataFrame from transcripts and save CSV if requested."""
    metadata = load_metadata(metadata_path)
    rows = gather_transcript_rows(transcripts_dir, metadata)
    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"CSV file created successfully: {output_path}")

    return df

# Drop missing lines and add stratify label for balanced splitting
def prepare_data_for_split(df):
    """Drop missing lines and add stratify label for balanced splitting."""
    df = df.dropna(subset=['Line']).copy()
    df['stratify_label'] = (
        df['Variety'].astype(str) + "_" +
        df['Sex'].astype(str) + "_" +
        df['Age group'].astype(str)
    )
    return df