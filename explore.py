# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Set nice style for ALL graphs
sns.set_style("whitegrid")

print("Setup complete! Ready to draw graphs ✅")


# ============================================
# GRAPH 1: Price Distribution Histogram
# ============================================
print("Drawing Graph 1...")

plt.figure(figsize=(10, 5))
plt.hist(df['Price'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Number of Houses')
plt.axvline(df['Price'].mean(),
            color='red',
            linestyle='--',
            label=f"Average: ${df['Price'].mean():.2f}")
plt.legend()
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()
print("Graph 1 done! ✅")

# ============================================
# GRAPH 2: Correlation Heatmap
# ============================================
print("Drawing Graph 2...")

plt.figure(figsize=(10, 8))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            square=True,
            linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Graph 2 done! ✅")

# ============================================
# GRAPH 3: Income vs Price Scatter Plot
# ============================================
print("Drawing Graph 3...")

plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['Price'],
            alpha=0.2,
            color='steelblue',
            edgecolors='none',
            s=10)
plt.title('Median Income vs House Price', fontsize=16)
plt.xlabel('Median Income (in $10,000s)')
plt.ylabel('House Price (in $100,000s)')
plt.tight_layout()
plt.savefig('income_vs_price.png')
plt.show()
print("Graph 3 done! ✅")

# ============================================
# GRAPH 4: Boxplot - Spotting Outliers
# ============================================
print("Drawing Graph 4...")

plt.figure(figsize=(12, 5))
df[['MedInc', 'HouseAge', 'AveRooms', 'Price']].boxplot(
    patch_artist=True,
    grid=True)
plt.title('Boxplot — Detecting Outliers', fontsize=16)
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('boxplot_outliers.png')
plt.show()
print("Graph 4 done! ✅")

print("\n All 4 graphs complete and saved! 🎉")