"""
Utility functions for data preprocessing and text cleaning
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load news article dataset from CSV
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe with Article and Category columns
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} articles, {df.shape[1]} columns")
    return df


def clean_text(text: str) -> str:
    """
    Clean article text
    
    Args:
        text (str): Raw article text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    
    return text


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Article'], keep='first')
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate articles")
    else:
        print("No duplicates found")
    
    return df


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Check for missing values in dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary with missing value counts
    """
    missing = df.isnull().sum()
    missing_dict = {col: int(count) for col, count in missing[missing > 0].items()}
    
    if missing_dict:
        print(f"Missing values found: {missing_dict}")
    else:
        print("No missing values found")
    
    return missing_dict


def get_article_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate article length statistics
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Statistics including mean, median, min, max lengths
    """
    df['word_count'] = df['Article'].apply(lambda x: len(str(x).split()))
    
    stats = {
        'mean_length': df['word_count'].mean(),
        'median_length': df['word_count'].median(),
        'min_length': df['word_count'].min(),
        'max_length': df['word_count'].max(),
        'std_length': df['word_count'].std()
    }
    
    return stats


def check_class_balance(df: pd.DataFrame) -> pd.Series:
    """
    Check class distribution in dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Value counts of categories
    """
    class_dist = df['Category'].value_counts().sort_index()
    print("Class Distribution:")
    print(class_dist)
    
    # Check if balanced
    if class_dist.std() == 0:
        print("Dataset is perfectly balanced")
    else:
        print("Dataset shows class imbalance")
    
    return class_dist


def prepare_data(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """
    Complete data preparation pipeline
    
    Args:
        filepath (str): Path to dataset CSV
        
    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned dataframe and statistics
    """
    print("=" * 50)
    print("Starting Data Preparation...")
    print("=" * 50)
    
    # Load data
    df = load_data(filepath)
    print()
    
    # Check missing values
    check_missing_values(df)
    print()
    
    # Remove duplicates
    df = remove_duplicates(df)
    print()
    
    # Get statistics
    stats = get_article_statistics(df)
    print(f"\nArticle Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}")
    print()
    
    # Check class balance
    check_class_balance(df)
    print()
    
    print("=" * 50)
    print("Data Preparation Complete!")
    print("=" * 50)
    
    return df, stats


if __name__ == "__main__":
    # Example usage
    df, stats = prepare_data('data/article_data.csv')
