# model.py
# ============================================
# PHASE 4: Build Linear Regression Model
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ---- LOAD DATA ----
print("Loading data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print("Data loaded! ✅")

# ---- STEP 4A: Separate Features and Target ----
# X = Features (inputs) — what we USE to predict
# y = Target (output) — what we WANT to predict

X = df.drop('Price', axis=1)   # Everything EXCEPT price
y = df['Price']                 # Only the price column

print("\n--- Features (X) shape:", X.shape)
print("--- Target (y) shape:", y.shape)
print("\nFeature columns:", X.columns.tolist())


# ---- STEP 4B: Train Test Split ----
print("\n--- Splitting data into Train and Test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,              # Features
    y,              # Target
    test_size=0.2,  # 20% for testing
    random_state=42 # Fixes randomness — same split every time
)

print(f"Training set size:  {X_train.shape[0]} houses")
print(f"Testing set size:   {X_test.shape[0]} houses")
print("Split complete! ✅")


# ---- STEP 4C: Create and Train the Model ----
print("\n--- Training the model...")

# Step 1: Create a blank Linear Regression model
model = LinearRegression()

# Step 2: Train it! (This is where learning happens!)
model.fit(X_train, y_train)

print("Model trained successfully! ✅")

# See what the model learned
print(f"\nModel Intercept (b): {model.intercept_:.4f}")
print("\nFeature Coefficients (m values):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:15} → {coef:.4f}")

    # ---- STEP 4D: Make Predictions ----
print("\n--- Making Predictions...")

# Model predicts prices for ALL test houses
y_predicted = model.predict(X_test)

print("Predictions made! ✅")
print(f"\nTotal houses predicted: {len(y_predicted)}")

# Compare first 5 predictions vs real prices
print("\n--- First 5 Predictions vs Real Prices ---")
print(f"{'House':<8} {'Real Price':>12} {'Predicted':>12} {'Difference':>12}")
print("-" * 48)

for i in range(5):
    real      = y_test.iloc[i] * 100000
    predicted = y_predicted[i]  * 100000
    diff      = abs(real - predicted)
    print(f"House {i+1:<3} ${real:>11,.0f} ${predicted:>11,.0f} ${diff:>11,.0f}")

    # ---- STEP 4E: Evaluate Model Accuracy ----
print("\n--- Evaluating Model Accuracy...")

# Calculate all 3 metrics
mae  = metrics.mean_absolute_error(y_test, y_predicted)
mse  = metrics.mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2   = metrics.r2_score(y_test, y_predicted)

print("\n╔══════════════════════════════════════╗")
print("║       MODEL ACCURACY REPORT          ║")
print("╠══════════════════════════════════════╣")
print(f"║  MAE  (Avg error amount)  : ${mae*100000:>8,.0f}  ║")
print(f"║  RMSE (Punishes big err)  : ${rmse*100000:>8,.0f}  ║")
print(f"║  R²   (Overall score)     :   {r2:.4f}  ║")
print("╠══════════════════════════════════════╣")

if r2 >= 0.8:
    print("║  Rating: EXCELLENT! 🌟               ║")
elif r2 >= 0.6:
    print("║  Rating: GOOD! ✅                    ║")
elif r2 >= 0.4:
    print("║  Rating: AVERAGE 📈 (improvable)     ║")
else:
    print("║  Rating: NEEDS IMPROVEMENT 🔧        ║")

print("╚══════════════════════════════════════╝")

# ---- STEP 4F: Visualize Predictions ----
print("\n--- Drawing prediction graph...")

plt.figure(figsize=(10, 6))

# Plot real prices vs predicted prices as dots
plt.scatter(y_test,         # X axis = Real prices
            y_predicted,    # Y axis = Predicted prices
            alpha=0.3,
            color='steelblue',
            s=10,
            label='Predictions')

# Draw a perfect prediction line (diagonal)
# If all predictions were perfect, all dots would sit ON this line!
min_val = min(y_test.min(), y_predicted.min())
max_val = max(y_test.max(), y_predicted.max())

plt.plot([min_val, max_val],
         [min_val, max_val],
         color='red',
         linewidth=2,
         linestyle='--',
         label='Perfect Prediction Line')

# Labels
plt.title('Real Prices vs Predicted Prices', fontsize=16)
plt.xlabel('Real Price (in $100,000s)')
plt.ylabel('Predicted Price (in $100,000s)')
plt.legend()
plt.tight_layout()
plt.savefig('predictions_vs_real.png')
plt.show()

print("Prediction graph saved! ✅")
print("\n🎉 Phase 4 Complete!")



