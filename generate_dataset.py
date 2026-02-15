"""
Generate a sample dataset for testing the ML classification models.
This creates a dataset with >= 500 rows and >= 12 features for binary classification.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification dataset
# 1000 samples, 15 features (more than the required 12)
X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=10,
    n_redundant=3,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=[0.6, 0.4],
    flip_y=0.05,
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i+1}' for i in range(15)]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Add some categorical features for variety
df['category_A'] = np.random.choice(['Group1', 'Group2', 'Group3'], size=1000)
df['category_B'] = np.random.choice(['TypeX', 'TypeY'], size=1000)

# Save to CSV
df.to_csv('sample_dataset.csv', index=False)

print("Sample dataset generated successfully!")
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Features: {df.shape[1] - 1}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['target'].value_counts())
