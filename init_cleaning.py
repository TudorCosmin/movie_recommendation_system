import pandas as pd
import os

def clean_movie_data(movies_filepath: str, output_filepath: str) -> None:
    """
    Cleans the movie CSV file by filtering out rows with non-integer IDs and saves the cleaned data to a new file. 
    If the output file already exists, the function does nothing.

    Args:
        movies_filepath: Path to the input CSV file containing movie data.
        output_filepath: Path to save the cleaned CSV file.

    Returns:
        None
    """
    if os.path.exists(output_filepath):
        return

    df = pd.read_csv(movies_filepath, low_memory=False)
    
    def is_integer(value):
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    df = df[df["id"].apply(is_integer)]

    df.to_csv(output_filepath, index=False)
    print(f"Wrote cleaned CSV to {output_filepath} with {len(df)} rows.")