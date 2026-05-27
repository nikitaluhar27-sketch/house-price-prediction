# save_model.py
# ============================================
# Save the trained model to a file
# ============================================

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle  # pickle = saves Python objects to files

# Load data
print("Loading data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Prepare features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to a file using pickle
print("Saving model...")
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as house_model.pkl ✅")
print("You can now use this model in the web app!")