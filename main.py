# main.py
# ============================================
# HOUSE PRICE PREDICTION - Step 2: Load Data
# ============================================

# Import the libraries we need
import pandas as pd
import numpy as np

# Scikit-learn has free datasets built in!
from sklearn.datasets import fetch_california_housing

# ---- LOAD THE DATASET ----
print("Loading dataset...")
housing = fetch_california_housing()

# Convert it into a pandas DataFrame (like an Excel table)
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target column (house prices) to our table
df['Price'] = housing.target

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)



# ---- EXPLORE THE DATA ----

# Show first 5 rows (like peeking at the top of an Excel sheet)
print("\n--- First 5 rows of data ---")
print(df.head())

# Show column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Show basic statistics
print("\n--- Basic Statistics ---")
print(df.describe())



# ============================================
# PHASE 3: Data Exploration & Cleaning
# ============================================

# ---- CHECK FOR MISSING VALUES ----
print("=== MISSING VALUES CHECK ===")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())



# ---- CHECK DATA TYPES ----
print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ---- CHECK FOR OUTLIERS ----
print("\n=== PRICE STATISTICS ===")
print(f"Minimum Price:  ${df['Price'].min() * 100000:,.0f}")
print(f"Maximum Price:  ${df['Price'].max() * 100000:,.0f}")
print(f"Average Price:  ${df['Price'].mean() * 100000:,.0f}")
print(f"Median Price:   ${df['Price'].median() * 100000:,.0f}")





