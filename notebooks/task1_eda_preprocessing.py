"""
Task 1: Exploratory Data Analysis and Data Preprocessing

This script performs:
1. Download and loading of the CFPB complaint dataset
2. Exploratory Data Analysis
3. Data preprocessing and cleaning
4. Saving the cleaned dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def download_dataset():
    """
    Function to download the CFPB dataset.
    Note: This is a placeholder function. In practice, you would need to download
    the dataset from: https://drive.google.com/file/d/1MMmioXFFOVMIc7GTrXNefgXM6UiHuCZ8/view?usp=sharing
    """
    print("Please download the dataset from: https://drive.google.com/file/d/1MMmioXFFOVMIc7GTrXNefgXM6UiHuCZ8/view?usp=sharing")
    print("And place it in the data/raw/ directory as 'complaints.csv'")
    return None

def load_data(filepath):
    """
    Load the complaint dataset from the specified filepath
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the complaint dataset
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")

    # First few rows
    print(f"\nFirst 5 rows:\n{df.head()}")

    # Data types
    print(f"\nData types:\n{df.dtypes}")

    # Missing values
    print(f"\nMissing values before handling:\n{df.isnull().sum()}")

    # Basic statistics
    print(f"\nBasic statistics:\n{df.describe(include='all')}")

    # Distribution of complaints across products
    if 'Product' in df.columns or 'product' in df.columns:
        product_col = 'Product' if 'Product' in df.columns else 'product'
        print(f"\nDistribution of complaints across {product_col}s:")
        print(df[product_col].value_counts())

        # Plot distribution
        plt.figure(figsize=(12, 6))
        df[product_col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of Complaints Across {product_col}s')
        plt.xlabel(product_col)
        plt.ylabel('Number of Complaints')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('complaints_product_distribution.png')
        plt.show()

    # Consumer complaint narrative analysis
    narrative_col = None
    if 'Consumer complaint narrative' in df.columns:
        narrative_col = 'Consumer complaint narrative'
    elif 'consumer_complaint_narrative' in df.columns:
        narrative_col = 'consumer_complaint_narrative'
    elif 'narrative' in df.columns:
        narrative_col = 'narrative'

    if narrative_col:
        print(f"\nAnalyzing {narrative_col}:")

        # Count non-null narratives
        non_null_narratives = df[narrative_col].notna().sum()
        print(f"Number of complaints with narratives: {non_null_narratives}")
        print(f"Number of complaints without narratives: {df.shape[0] - non_null_narratives}")

        # Calculate word counts
        df['word_count'] = df[narrative_col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )

        print(f"\nWord count statistics for narratives:")
        print(f"Mean word count: {df['word_count'].mean():.2f}")
        print(f"Median word count: {df['word_count'].median():.2f}")
        print(f"Min word count: {df['word_count'].min()}")
        print(f"Max word count: {df['word_count'].max()}")

        # Plot word count distribution
        plt.figure(figsize=(12, 6))
        plt.hist(df['word_count'], bins=50, edgecolor='black')
        plt.title('Distribution of Word Counts in Consumer Complaint Narratives')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('narrative_word_count_distribution.png')
        plt.show()

        # Show examples of short and long narratives
        print(f"\nExample of a short narrative (word count: {df['word_count'].min()}):")
        if df['word_count'].min() > 0:
            short_idx = df['word_count'].idxmin()
            print(f"'{df.loc[short_idx, narrative_col]}'")

        print(f"\nExample of a long narrative (word count: {df['word_count'].max()}):")
        long_idx = df['word_count'].idxmax()
        print(f"'{df.loc[long_idx, narrative_col]}'")

    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    print(f"\nHandling missing values...")

    # For categorical columns, fill missing values with 'Unknown' or 'Not Specified'
    categorical_columns = ['Sub-product', 'Sub-issue', 'Company public response', 'State',
                          'ZIP code', 'Tags', 'Consumer consent provided?', 'Consumer disputed?',
                          'Company response to consumer']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # For the 'Issue' column, fill with mode (most frequent value) since there are typically few missing values
    if 'Issue' in df.columns:
        df['Issue'] = df['Issue'].fillna(df['Issue'].mode()[0] if not df['Issue'].mode().empty else 'Unknown')

    # For 'Consumer complaint narrative', we'll keep NaN values as they will be handled during filtering
    # since we only want records with narratives for our RAG system

    # Display missing values after handling
    print(f"\nMissing values after handling:\n{df.isnull().sum()}")

    return df

def filter_and_clean_data(df):
    """
    Filter and clean the dataset according to project requirements
    """
    print("\n" + "="*50)
    print("FILTERING AND CLEANING DATA")
    print("="*50)
    
    initial_count = df.shape[0]
    print(f"Initial number of complaints: {initial_count}")
    
    # Identify the correct column names for product and narrative
    product_col = None
    narrative_col = None
    
    for col in df.columns:
        if col.lower() in ['product', 'product_name', 'producttype']:
            product_col = col
        elif col.lower() in ['consumer complaint narrative', 'consumer_complaint_narrative', 'narrative', 'complaint narrative']:
            narrative_col = col
    
    print(f"Identified product column: {product_col}")
    print(f"Identified narrative column: {narrative_col}")
    
    if product_col is None:
        print("ERROR: Could not identify product column. Available columns:", df.columns.tolist())
        return df
    
    # Filter for the specified products
    target_products = [
        'Credit card', 'Credit Card', 'credit card',
        'Personal loan', 'Personal Loan', 'personal loan',
        'Savings account', 'Savings Account', 'savings account',
        'Money transfers', 'Money Transfers', 'money transfers'
    ]
    
    df_filtered = df[df[product_col].isin(target_products)].copy()
    print(f"After filtering for target products: {df_filtered.shape[0]} complaints")
    
    # Remove records with empty narratives if narrative column exists
    if narrative_col:
        df_filtered = df_filtered[df_filtered[narrative_col].notna()].copy()
        df_filtered = df_filtered[df_filtered[narrative_col].str.strip() != ''].copy()
        print(f"After removing records with empty narratives: {df_filtered.shape[0]} complaints")
    
    # Clean the text narratives
    if narrative_col:
        # Lowercase text
        df_filtered[narrative_col] = df_filtered[narrative_col].str.lower()
        
        # Remove special characters (keeping letters, numbers, spaces, and basic punctuation)
        import re
        df_filtered[narrative_col] = df_filtered[narrative_col].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)\'\"]', ' ', str(x)) if pd.notna(x) else x
        )
        
        # Remove extra whitespace
        df_filtered[narrative_col] = df_filtered[narrative_col].apply(
            lambda x: ' '.join(str(x).split()) if pd.notna(x) else x
        )
    
    print(f"Final filtered dataset shape: {df_filtered.shape}")
    print(f"Removed {initial_count - df_filtered.shape[0]} complaints ({((initial_count - df_filtered.shape[0])/initial_count)*100:.2f}%)")
    
    return df_filtered

def save_filtered_data(df, filepath):
    """
    Save the filtered and cleaned dataset
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"Filtered dataset saved to {filepath}")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

def main():
    """
    Main function to execute Task 1
    """
    print("Starting Task 1: Exploratory Data Analysis and Data Preprocessing")
    
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Download dataset (manual step)
    download_dataset()
    
    # For demonstration purposes, let's create a sample dataset
    # In real implementation, you would load the actual dataset
    print("\nCreating a sample dataset for demonstration...")
    
    # Sample data based on the CFPB dataset structure
    sample_data = {
        'Date received': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'Product': np.random.choice([
            'Credit card', 'Personal loan', 'Savings account', 'Money transfers'
        ], size=1000),
        'Sub-product': [''] * 1000,
        'Issue': np.random.choice([
            'Billing dispute', 'Interest rate', 'Account opening', 'Transfer issues'
        ], size=1000),
        'Sub-issue': [''] * 1000,
        'Consumer complaint narrative': [
            f"This is a sample complaint narrative about {np.random.choice(['billing', 'interest', 'fees', 'service'])} issues. " * np.random.randint(1, 5)
            for _ in range(1000)
        ] * 1  # Just repeat for demo
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Perform EDA on sample data
    df_eda = perform_eda(df_sample)

    # Handle missing values
    df_with_missing_handled = handle_missing_values(df_eda)

    # Filter and clean the sample data
    df_filtered = filter_and_clean_data(df_with_missing_handled)
    
    # Save the filtered data
    save_filtered_data(df_filtered, "data/filtered_complaints.csv")
    
    print("\nTask 1 completed successfully!")
    
    return df_filtered

if __name__ == "__main__":
    df_result = main()