
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import load_data, clean_data, feature_engineering, create_lags_and_rolling
from src.forecasting import train_and_evaluate, predict_future
from src.inventory import run_simulation, calculate_eoq_rop

st.set_page_config(page_title="Expiry-Aware Inventory Management", layout="wide")

# --- Custom CSS for "Rich Aesthetics" ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1f2937;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #374151;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_and_prep_data():
    train, test, expiry = load_data()
    train = clean_data(train)
    
    # Merge expiry
    train = train.merge(expiry, on='item', how='left')
    train['expiry_days'] = train['expiry_days'].fillna(30) # Default if missing
    
    # Feature Engineering (Global)
    # Be careful with memory. 1.2M rows is fine.
    train = feature_engineering(train)
    train = create_lags_and_rolling(train)
    
    return train, test, expiry

with st.spinner("Loading Data..."):
    df_train, df_test, df_expiry = load_and_prep_data()

# --- Sidebar ---
st.sidebar.title("Configuration")

# Store/Item Selection
stores = sorted(df_train['store'].unique())
selected_store = st.sidebar.selectbox("Select Store", stores)

items_in_store = df_train[df_train['store'] == selected_store]['item'].unique()
selected_item = st.sidebar.selectbox("Select Item", sorted(items_in_store))

# Filter Data
item_data = df_train[(df_train['store'] == selected_store) & (df_train['item'] == selected_item)].copy()
item_data = item_data.sort_values('date')

# Expiry for this item
item_expiry = item_data['expiry_days'].iloc[0] if not item_data.empty else 30
st.sidebar.markdown(f"**Item Expiry:** {int(item_expiry)} days")

st.sidebar.subheader("Inventory Parameters")
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10.0, 200.0, 50.0)
holding_cost = st.sidebar.slider("Holding Cost ($/unit/day)", 0.01, 1.0, 0.1)
shortage_cost = st.sidebar.slider("Shortage Cost ($/unit)", 1.0, 50.0, 10.0)
waste_cost = st.sidebar.slider("Waste Cost ($/unit)", 0.1, 20.0, 5.0)
lead_time = st.sidebar.slider("Lead Time (Days)", 1, 30, 7)
target_service_z = st.sidebar.slider("Target Service Level Z", 1.0, 3.0, 1.96) # 1.96 = 97.5%

# --- Forecasting ---
st.header("1. Forecasting")

# Train Model (Global or Local? - For demo speed, let's train on this Item history + some context if needed, 
# but usually we want global. Let's start with a simple per-series train or global. 
# Global is better for accuracy but slower. Let's do per-series for instant feedback in this UI demo if dataset is huge,
# BUT the document says "LightGBM ... categorical features for store/item". Implies Global.
# I will try to cache the Global Model.

@st.cache_resource
def get_trained_model(data):
    # This might fail on memory if not careful, but let's try.
    model, rmsle, baseline_rmsle, feats = train_and_evaluate(data)
    return model, rmsle, baseline_rmsle, feats

if st.sidebar.button("Retrain Model"):
    st.cache_resource.clear()

with st.spinner("Training/Loading Forecast Model..."):
    # To save time, we can sample or just use the whole thing.
    # For this specific item view, we might want to see the global performance.
    
    # NOTE: Training on 1.2M rows might take 30s-1m.
    model, val_rmsle, base_rmsle, feature_cols = get_trained_model(df_train)

st.success(f"Model Trained! Validation RMSLE: {val_rmsle:.4f} (Baseline: {base_rmsle:.4f})")

# Predict for the specific item (last 90 days of train as 'simulation' period, or future?)
# Text says "last 3 months validation set". Let's visualize the validation period for this item.
val_start_date = item_data['date'].max() - pd.DateOffset(months=3)
sim_data = item_data[item_data['date'] > val_start_date].copy()

# Generate predictions
# We need to ensure sim_data has the features.
pred_sales = model.predict(sim_data[feature_cols])
sim_data['forecast'] = np.clip(pred_sales, 0, None)

# Plot Forecast vs Actual
fig_forecast = px.line(sim_data, x='date', y=['sales', 'forecast'], title=f"Sales vs Forecast (Store {selected_store}, Item {selected_item})",
                       color_discrete_map={'sales': '#1f77b4', 'forecast': '#ff7f0e'})
st.plotly_chart(fig_forecast, use_container_width=True)


# --- Inventory Simulation ---
st.header("2. Inventory Simulation & Optimization")

# Calculate EOQ / ROP based on Forecast Statistics (simulated 'planning' based on forecast mean/std)
forecast_mean = sim_data['forecast'].mean()
forecast_std = sim_data['forecast'].std()

eoq, rop = calculate_eoq_rop(forecast_mean, forecast_std, ordering_cost, holding_cost, lead_time, target_service_z)

col1, col2 = st.columns(2)
col1.metric("Calculated EOQ", f"{eoq:.2f}")
col2.metric("Calculated ROP", f"{rop:.2f}")

# Run Simulation
sim_params = {
    'ordering_cost': ordering_cost,
    'holding_cost': holding_cost,
    'shortage_cost': shortage_cost,
    'waste_cost': waste_cost,
    'lead_time': lead_time,
    'eoq': eoq,
    'rop': rop
}

# We simulate using ACTUAL demand to see how our planning (based on forecast) performed?
# Or we simulate using FORECAST? 
# Real world: You plan with forecast, reality is actuals.
# So we feed 'sales' (actuals) as the demand stream to the simulator, 
# but the simulator parameters (EOQ, ROP) were derived from the Forecast. i.e. "Inventory Logic"
sim_results = run_simulation(sim_data['sales'].values, initial_stock=rop, expiry_days=item_expiry, params=sim_params)

# Metrics Display
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='metric-card'><div class='metric-label'>Service Level</div><div class='metric-value'>{sim_results['service_level']*100:.1f}%</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><div class='metric-label'>Stockouts</div><div class='metric-value'>{sim_results['stockout_days']}</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><div class='metric-label'>Waste</div><div class='metric-value'>{sim_results['waste_units']:.0f}</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card'><div class='metric-label'>Total Cost</div><div class='metric-value'>${sim_results['total_cost']:,.0f}</div></div>", unsafe_allow_html=True)

# Plot Inventory
sim_df = sim_results['details']
sim_df['date'] = sim_data['date'].values
sim_df['rop'] = rop

fig_inv = go.Figure()
fig_inv.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['stock'], fill='tozeroy', name='Inventory Level', line=dict(color='#10b981')))
fig_inv.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['rop'], name='Reorder Point', line=dict(color='red', dash='dash')))
fig_inv.update_layout(title="Inventory Levels over Time", xaxis_title="Date", yaxis_title="Units")
st.plotly_chart(fig_inv, use_container_width=True)

# --- What-If Analysis ---
st.header("3. What-If Analysis")
if st.button("Run Sensitivity Analysis (Grid Search on Z)"):
    z_values = np.linspace(1.28, 3.0, 10) # 90% to 99.9%
    results_list = []
    
    progress_bar = st.progress(0)
    
    for i, z in enumerate(z_values):
        # Recalculate ROP based on z
        _, r = calculate_eoq_rop(forecast_mean, forecast_std, ordering_cost, holding_cost, lead_time, z)
        p = sim_params.copy()
        p['rop'] = r
        res = run_simulation(sim_data['sales'].values, initial_stock=r, expiry_days=item_expiry, params=p)
        results_list.append({
            'Z-Score': z,
            'Service Level': res['service_level'],
            'Total Cost': res['total_cost'],
            'Waste': res['waste_units']
        })
        progress_bar.progress((i + 1) / len(z_values))
        
    res_df = pd.DataFrame(results_list)
    
    # Plot Cost vs Service Level
    fig_tradeoff = px.scatter(res_df, x='Service Level', y='Total Cost', size='Waste', color='Z-Score',
                              title="Tradeoff: Cost vs Service Level (Size = Waste)",
                              labels={'Service Level': 'Service Level', 'Total Cost': 'Total Cost'})
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    st.dataframe(res_df.style.highlight_min(subset=['Total Cost'], color='lightgreen'))

# --- Exports ---
st.header("4. Export Results")
col_ex1, col_ex2 = st.columns(2)

# Time Series CSV
csv_ts = sim_df.to_csv(index=False).encode('utf-8')
col_ex1.download_button(
    label="Download Timeseries CSV",
    data=csv_ts,
    file_name=f'inventory_timeseries_store{selected_store}_item{selected_item}.csv',
    mime='text/csv',
)

# Summary CSV (Current Simulation Params & Metrics)
summary_data = {
    'Store': [selected_store],
    'Item': [selected_item],
    'Params': [str(sim_params)],
    'Service Level': [sim_results['service_level']],
    'Stockouts': [sim_results['stockout_days']],
    'Waste': [sim_results['waste_units']],
    'Total Cost': [sim_results['total_cost']]
}
csv_sum = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
col_ex2.download_button(
    label="Download Summary CSV",
    data=csv_sum,
    file_name=f'inventory_summary_store{selected_store}_item{selected_item}.csv',
    mime='text/csv',
)


st.markdown("---")
st.markdown("End of Report")
