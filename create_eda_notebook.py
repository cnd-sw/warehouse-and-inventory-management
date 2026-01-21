
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Data Analytics & EDA: Warehouse Inventory Data\n\nThis notebook performs comprehensive exploratory data analysis on the provided datasets."),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configs
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
%matplotlib inline"""),

    nbf.v4.new_markdown_cell("## 1. Load Data"),
    
    nbf.v4.new_code_cell("""# Define paths
DATA_DIR = 'data/'

# Load CSVs
train_df = pd.read_csv(f"{DATA_DIR}train.csv", parse_dates=['date'])
test_df = pd.read_csv(f"{DATA_DIR}test.csv", parse_dates=['date'])
expiry_df = pd.read_csv(f"{DATA_DIR}expiry_mapping.csv")

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"Expiry Shape: {expiry_df.shape}")"""),

    nbf.v4.new_markdown_cell("## 2. Basic Data Inspection"),
    
    nbf.v4.new_code_cell("""display(train_df.head())
print(train_df.info())
display(train_df.describe())"""),

    nbf.v4.new_markdown_cell("## 3. Univariate Analysis: Sales"),
    
    nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.histplot(train_df['sales'], bins=50, kde=True, color='blue')
plt.title('Distribution of Daily Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()"""),

    nbf.v4.new_markdown_cell("## 4. Time Series Analysis"),

    nbf.v4.new_code_cell("""# Sales over Time (Aggregated across all stores/items)
daily_sales = train_df.groupby('date')['sales'].sum().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(daily_sales['date'], daily_sales['sales'])
plt.title('Total Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()"""),

    nbf.v4.new_markdown_cell("## 5. Seasonality Analysis"),

    nbf.v4.new_code_cell("""# Feature Engineering for Plotting
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['day_name'] = train_df['date'].dt.day_name()

# Monthly Seasonality
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='sales', data=train_df)
plt.title('Sales Distribution by Month')
plt.show()

# Weekly Seasonality
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(12, 6))
sns.boxplot(x='day_name', y='sales', data=train_df, order=order_days)
plt.title('Sales Distribution by Day of Week')
plt.show()"""),

    nbf.v4.new_markdown_cell("## 6. Store & Item Analysis"),

    nbf.v4.new_code_cell("""# Total Sales by Store
store_sales = train_df.groupby('store')['sales'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='store', y='sales', data=store_sales, palette='viridis')
plt.title('Total Sales by Store')
plt.show()

# Total Sales by Item
item_sales = train_df.groupby('item')['sales'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='item', y='sales', data=item_sales.head(20), palette='magma') # Top 20 items
plt.title('Top 20 Items by Total Sales')
plt.xlabel('Item ID')
plt.show()"""),

    nbf.v4.new_markdown_cell("## 7. Expiry & Shelf Life Analysis"),

    nbf.v4.new_code_cell("""# Merge Shelf Life Data
expiry_df = expiry_df.rename(columns={'item': 'item_id'}) # Ensure loose match if column names differ
# Actually, let's check column compatibility:
# data/expiry_mapping.csv usually has 'item' or 'item_id'
# Let's re-read to be sure or just display it first.
display(expiry_df.head())

plt.figure(figsize=(10, 6))
sns.histplot(expiry_df['expiry_days'], bins=20, kde=True, color='green')
plt.title('Distribution of Shelf Life (Days)')
plt.xlabel('Days')
plt.show()"""),

    nbf.v4.new_markdown_cell("## 8. Correlation Analysis"),

    nbf.v4.new_code_cell("""# Pivot table to check correlation between stores (Do stores sell similarly?)
pivot_sales = train_df.pivot_table(index='date', columns='store', values='sales', aggfunc='sum')
corr_matrix = pivot_sales.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sales Correlation between Stores')
plt.show()""")
]

with open('warehouse_analytics.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 'warehouse_analytics.ipynb' created successfully.")
