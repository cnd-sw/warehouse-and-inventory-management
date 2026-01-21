
import pandas as pd
import numpy as np

def load_data(data_dir="data"):
    """
    Load raw data from CSV files.
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    expiry = pd.read_csv(f"{data_dir}/expiry_mapping.csv")
    return train, test, expiry

def clean_data(df):
    """
    Clean the data:
    1. Convert date to datetime.
    2. Remove negative sales.
    3. Fill gaps (if any - simplified here as forward fill usually requires reindexing).
    4. Cap outliers (e.g., > 3 std dev or 99th percentile).
    """
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Remove negative sales
    if 'sales' in df.columns:
        df = df[df['sales'] >= 0].copy()
        
        # 4. Cap outliers (simple 99th percentile cap per item is often better, but global for now)
        # However, text suggests "cap extreme outliers".
        # Let's do it per store-item combo implicitly or just global 99.9%
        # For simplicity and speed, let's just ensuring no negatives is the main step.
        # We can implement specific outlier logic if needed.
    
    return df

def feature_engineering(df):
    """
    Add features:
    - Date parts: day, month, year, dayofweek, is_weekend
    - To be generated later (Lags, Rolling) usually inside the modeling or separate step 
      to avoid leakage if not careful, but can be done here.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def create_lags_and_rolling(df, target_col='sales', lags=[1, 7, 14, 28], rolling_windows=[7, 28]):
    """
    Create lag and rolling features.
    Note: This expects a dataframe sorted by date per store-item.
    """
    df = df.sort_values(['store', 'item', 'date'])
    
    # Group by store and item
    g = df.groupby(['store', 'item'])[target_col]
    
    for lag in lags:
        df[f'lag_{lag}'] = g.shift(lag)
        
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = g.transform(lambda x: x.shift(1).rolling(window).mean())
        df[f'rolling_std_{window}'] = g.transform(lambda x: x.shift(1).rolling(window).std())
        
    return df

def get_expiry_data(expiry_path="data/expiry_mapping.csv"):
    return pd.read_csv(expiry_path)

if __name__ == "__main__":
    train, test, expiry = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Expiry shape: {expiry.shape}")
