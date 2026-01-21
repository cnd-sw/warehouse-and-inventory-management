
import sys
import pandas as pd
import numpy as np
import traceback

print("1. Importing modules...")
try:
    from src.data_processing import load_data, clean_data, feature_engineering, create_lags_and_rolling
    from src.forecasting import train_and_evaluate
    from src.inventory import run_simulation, calculate_eoq_rop
    print("   Modules imported successfully.")
except Exception as e:
    print(f"   FATAL: Module import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n2. Loading and Prepping Data...")
try:
    train, test, expiry = load_data()
    print(f"   Loaded train: {train.shape}, expiry: {expiry.shape}")
    
    train = clean_data(train)
    
    # Check for expected columns
    expected_cols = ['date', 'store', 'item', 'sales']
    missing = [c for c in expected_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in clean data: {missing}")
        
    train = train.merge(expiry, on='item', how='left')
    train['expiry_days'] = train['expiry_days'].fillna(30)
    
    # Feature Engineering
    print("   Running Feature Engineering (this might take a moment)...")
    train = feature_engineering(train)
    train = create_lags_and_rolling(train)
    print("   Feature Engineering Complete.")
    print(f"   Train shape after FE: {train.shape}")
    
except Exception as e:
    print(f"   FATAL: Data Prep failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing Forecasting Logic (Sample)...")
try:
    # Filter for just one item-store to speed up test
    sample_sub = train[(train['store'] == 1) & (train['item'] == 1)].copy()
    if sample_sub.empty:
        # fallback if store 1 item 1 doesn't exist
        sample_sub = train.iloc[:1000].copy()
        print("   Warning: Store 1 Item 1 not found, using first 1000 rows.")
    
    print(f"   Training on sample subset: {sample_sub.shape}")
    model, val_rmsle, base_rmsle, feats = train_and_evaluate(sample_sub)
    print(f"   Model trained. RMSLE: {val_rmsle}")

except Exception as e:
    print(f"   FATAL: Forecasting failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing Simulation Logic...")
try:
    # Use valid data
    sim_data = sample_sub.iloc[-100:].copy() # last 100 days
    # Fake forecast for simulation input
    sim_data['forecast'] = sim_data['sales'].mean() 
    
    means = sim_data['forecast'].mean()
    stds = sim_data['forecast'].std() if len(sim_data) > 1 else 1.0
    
    eoq, rop = calculate_eoq_rop(means, stds, ordering_cost=50, holding_cost=0.1, lead_time=7)
    print(f"   EOQ: {eoq}, ROP: {rop}")
    
    sim_params = {
        'ordering_cost': 50,
        'holding_cost': 0.1,
        'shortage_cost': 10,
        'waste_cost': 5,
        'lead_time': 7,
        'eoq': eoq,
        'rop': rop
    }
    
    res = run_simulation(sim_data['sales'].values, initial_stock=rop, expiry_days=30, params=sim_params)
    print(f"   Simulation Result Service Level: {res['service_level']}")
    
except Exception as e:
    print(f"   FATAL: Simulation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nSUCCESS: All modules passed smoke test.")
