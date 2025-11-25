import pandas as pd

def tokenize_line(line):
    """
    Tokenize a single cleaned line into a list of words (tokens).
    """
    return line.split()


def tokenize_dataframe(df, text_column="line"):
    """
    Tokenize a dataframe column containing cleaned text.

    Args:
        df (pd.DataFrame): DataFrame with a column of cleaned text.
        text_column (str): Name of the column to tokenize.

    Returns:
        pd.DataFrame: New DataFrame with an added 'tokens' column.
    """
    df = df.copy()
    df["tokens"] = df[text_column].apply(tokenize_line)
    return df


if __name__ == "__main__":
    # Example usage
    csv_path = "data/clean/cleaned_data.csv"
    df = pd.read_csv(csv_path)
    df_tokenized = tokenize_dataframe(df)
    print(df_tokenized.head())
