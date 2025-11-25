import spacy
import pandas as pd

# Load Spanish language model (install first with: python -m spacy download es_core_news_sm)
nlp = spacy.load("es_core_news_sm")


def pos_tag_tokens(tokens):
    """
    Convert a list of tokens into a POS tag sequence.

    Args:
        tokens (list of str): Tokenized words from a cleaned line.

    Returns:
        list of str: POS tags corresponding to each token.
    """
    doc = nlp(" ".join(tokens))
    return [token.pos_ for token in doc]


def pos_tag_dataframe(df, tokens_column="tokens"):
    """
    Apply POS tagging to a dataframe column of token lists.

    Args:
        df (pd.DataFrame): DataFrame with a column of token lists.
        tokens_column (str): Name of the column containing tokens.

    Returns:
        pd.DataFrame: New DataFrame with an added 'pos_sequence' column.
    """
    df = df.copy()
    df["pos_sequence"] = df[tokens_column].apply(pos_tag_tokens)
    return df


if __name__ == "__main__":
    # Example usage
    csv_path = "data/clean/cleaned_data.csv"
    df = pd.read_csv(csv_path)
    
    # Ensure tokenization column exists
    if "tokens" not in df.columns:
        from tokenizer import tokenize_dataframe
        df = tokenize_dataframe(df)
    
    df_pos = pos_tag_dataframe(df)
    print(df_pos.head())
