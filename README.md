# Expiry-Aware Inventory Management System

A comprehensive end-to-end solution for inventory forecasting and optimization, specifically designed to handle perishable goods using **FEFO (First-Expired-First-Out)** logic.

This project replicates the system described in the project documentation (`review_final.pdf`), integrating Machine Learning forecasting with traditional Inventory Operations methods.

## Features

### 1. Advanced Forecasting
*   **Model**: LightGBM (Gradient Boosting Machine).
*   **Target**: Daily Sales.
*   **Features**:
    *   Time-series features: Lags (1, 7, 14, 28 days), Rolling Means/Stds (7, 28 days).
    *   Calendar features: Day, Month, Year, Day of Week, Weekend Flag.
    *   Store/Item metadata.
*   **Validation**: Time-based validation split (last 3 months) measuring RMSLE (Root Mean Squared Logarithmic Error).

### 2. Inventory Simulation & Optimization
*   **Policies**:
    *   **EOQ (Economic Order Quantity)**: Optimizes order size to balance ordering and holding costs.
    *   **ROP (Reorder Point)**: Determines when to reorder based on lead time and safety stock.
*   **Expiry Logic**:
    *   Tracks inventory batches with specific expiration dates.
    *   Implements **FEFO** dispatching to minimize waste.
    *   Automatically removes expired stock and accounts for it as "Waste Cost".
*   **Costs**: Tracks Total Cost (Ordering + Holding + Shortage + Waste).

### 3. Interactive Dashboard
A Streamlit-based UI for decision support:
*   **Simulation**: Run inventory simulations with adjustable parameters (Costs, Lead Time, Service Level Z).
*   **Visualizations**:
    *   Sales vs. Forecast charts.
    *   Daily Inventory Levels vs. Reorder Point.
*   **What-If Analysis**: Sensitivity analysis grid search to optimize Service Level vs. Cost.
*   **Exports**: Download detailed simulation time-series and summary reports as CSV.

## Project Structure

```
├── data/                   # Dataset directory (train.csv, test.csv, expiry_mapping.csv)
├── src/
│   ├── data_processing.py  # Data loading, cleaning, and feature engineering
│   ├── forecasting.py      # LightGBM training and evaluation logic
│   ├── inventory.py        # Inventory simulation engine (EOQ, ROP, FEFO)
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Installation

1.  **Environment Setup**:
    Ensure you have Python 3.9+ or a Conda environment active (e.g., `ml310`).

    ```bash
    conda activate ml310
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Data**:
    Ensure the `data/` folder is populated with:
    *   `train.csv`: Historical sales data.
    *   `expiry_mapping.csv`: Shelf life info per item.
    *   `test.csv`: (Optional) Validation/Future frame.

## Usage

Run the dashboard application:

```bash
streamlit run app.py
```

The app will open in your browser (usually `http://localhost:8501`).

1.  **Select Store & Item** from the sidebar.
2.  **Adjust Inventory Parameters** (Costs, Lead Time) to match your scenario.
3.  **Train Model**: Click to retrain the LightGBM forecast on the fly.
4.  **Analyze**: View the Service Level, Stockouts, and Waste metrics.
5.  **What-If**: Click "Run Sensitivity Analysis" to find the optimal safety stock setting.
6.  **Export**: Download the results for offline analysis.

## Troubleshooting

### LightGBM on macOS (Apple Silicon)
If you encounter an error like `Library not loaded: @rpath/libomp.dylib`, it means `OpenMP` is missing or not linked. The most reliable fix is to install LightGBM via Conda, which handles the dependency correctly:

```bash
conda install -n ml310 -c conda-forge lightgbm
```

Or install `libomp` via Homebrew:

```bash
brew install libomp
```

## Methodology Notes

*   **Forecasting**: Uses RMSLE to handle the wide range of sales data and penalize under-forecasting effectively in log-space.

*   **Simulation**: The simulation runs on a daily step, strictly adhering to constraints (lead times, shelf life). It assumes orders are placed at the end of the day if the inventory position falls below ROP.
