
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from src.data_processing import feature_engineering, create_lags_and_rolling

def get_rmsle(y_true, y_pred):
    """
    Calculate RMSLE. Encapsulates log conversion logic.
    """
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))

def train_and_evaluate(df, val_months=3):
    """
    Train LightGBM model with time-based validation.
    """
    # Ensure optimized features
    df = feature_engineering(df)
    df = create_lags_and_rolling(df)
    
    # Drop rows with NaN due to lags
    df = df.dropna()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Validation split
    max_date = df['date'].max()
    split_date = max_date - pd.DateOffset(months=val_months)
    
    train_data = df[df['date'] <= split_date]
    val_data = df[df['date'] > split_date]
    
    feature_cols = [c for c in df.columns if c not in ['date', 'sales', 'id', 'year']] 
    # Excluding year to avoid trend overfitting if not capturing trend properly, 
    # though usually safer to keep or replace with trend index if non-stationary.
    # Text says "day-of-week, month, store and product IDs, weekend flag, expiry days".
    # We added expiry in main flow, need to ensure it's there.
    
    print(f"Features: {feature_cols}")
    
    X_train = train_data[feature_cols]
    y_train = train_data['sales']
    X_val = val_data[feature_cols]
    y_val = val_data['sales']
    
    # LightGBM Parameters (std essentials + tweaks if needed)
    params = {
        'objective': 'regression',
        'metric': 'rmse', # optimizing rmse on log-transformed target is rmsle, or use 'rmse' on raw and calc rmsle later
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Using raw sales, so metric is RMSE, but we evaluate RMSLE.
    # Often better to predict log(sales) for RMSLE optimization.
    # Let's try raw first as per standard unless specified. 
    # Actually, minimizing RMSE on log(y) is the RMSLE objective.
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Predictions
    preds = model.predict(X_val)
    rmsle = get_rmsle(y_val, preds)
    print(f"Validation RMSLE: {rmsle}")
    
    # Compare with Baseline (mean per item-store)
    # Baseline: Group Mean Predictor
    # "Baseline RMSLE (mean): 0.241"
    # We calculate the mean from TRAIN and applying to VAL
    means = train_data.groupby(['store', 'item'])['sales'].mean().reset_index()
    val_baseline = val_data.merge(means, on=['store', 'item'], how='left', suffixes=('', '_mean'))
    val_baseline['sales_mean'] = val_baseline['sales_mean'].fillna(train_data['sales'].mean())
    
    baseline_rmsle = get_rmsle(val_data['sales'], val_baseline['sales_mean'])
    print(f"Baseline RMSLE: {baseline_rmsle}")
    
    return model, rmsle, baseline_rmsle, feature_cols

def predict_future(model, future_df, feature_cols):
    """
    Predict for future data.
    """
    # Ensure features exist
    # Note: Lags for future need iterative prediction or availability.
    # For this simplified implementation (like Kaggle M5), 
    # if predicting far out, we need to generate features iteratively or use just calendar features + old lags.
    # Given "Sales Forecasting" usually implies iterative if lags used.
    # But usually 'test' file assumes we have necessary inputs.
    # If lags are used, we can only predict step-by-step or if test data has recent history attached.
    
    # For simplicity, assuming future_df comes with necessary columns or we generate them.
    # Here we just predict:
    return model.predict(future_df[feature_cols])

