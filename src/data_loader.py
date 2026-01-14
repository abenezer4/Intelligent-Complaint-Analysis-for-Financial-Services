import pandas as pd
import os

def load_filtered_data(filepath: str = "data/filtered_complaints.csv") -> pd.DataFrame:
    """
    Load the filtered and cleaned complaint dataset.
    """
    if not os.path.exists(filepath):
        # Check if it exists in a different relative path (e.g. from notebooks)
        if os.path.exists(f"../{filepath}"):
            filepath = f"../{filepath}"
        else:
            print(f"Warning: File {filepath} not found.")
            return None

    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
