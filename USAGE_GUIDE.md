# ML Classification Models - Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ch-premchand/cuddly-octo-adventure.git
cd cuddly-octo-adventure

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Dataset (Optional)

```bash
python generate_dataset.py
```

This creates `sample_dataset.csv` with:
- 1000 rows (exceeds 500 minimum)
- 17 features (exceeds 12 minimum)
- Binary classification target
- Mix of numeric and categorical features

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Using the Application

### Step 1: Upload Dataset
- Click **"Browse files"** in the sidebar
- Select a CSV file with:
  - At least 500 rows
  - At least 12 features (plus 1 target column)
  - Headers in the first row

### Step 2: Configure
- **Select Target Column**: Choose the column containing your labels/classes
- **Test Size**: Adjust the train/test split ratio (default: 20%)
- **Random State**: Set seed for reproducibility (default: 42)

### Step 3: Select Models
Choose which models to train (or keep all 6 selected):
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ k-Nearest Neighbors (kNN)
- ✅ Naive Bayes
- ✅ Random Forest
- ✅ XGBoost

### Step 4: Train Models
Click the **"Train Models"** button to start training.

### Step 5: View Results
For each model, you'll see:
- **6 Metrics**: Accuracy, AUC, Precision, Recall, F1 Score, MCC
- **Confusion Matrix**: Visual heatmap of predictions
- **Comparison Summary**: Table comparing all models
- **Best Model**: Automatically identified by F1 Score
- **Comparison Charts**: Bar charts for all metrics

## Understanding the Metrics

### Accuracy
Overall correctness of predictions.
- Range: 0-1 (higher is better)
- Formula: (TP + TN) / (TP + TN + FP + FN)

### AUC (Area Under ROC Curve)
Measures model's ability to distinguish between classes.
- Range: 0-1 (higher is better)
- 0.5 = random guessing, 1.0 = perfect classification

### Precision
Proportion of positive predictions that were correct.
- Range: 0-1 (higher is better)
- Formula: TP / (TP + FP)
- Important when false positives are costly

### Recall (Sensitivity)
Proportion of actual positives correctly identified.
- Range: 0-1 (higher is better)
- Formula: TP / (TP + FN)
- Important when false negatives are costly

### F1 Score
Harmonic mean of precision and recall.
- Range: 0-1 (higher is better)
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Balances precision and recall

### MCC (Matthews Correlation Coefficient)
Balanced measure even with imbalanced classes.
- Range: -1 to +1 (higher is better)
- +1 = perfect prediction
- 0 = random prediction
- -1 = total disagreement

## Model Descriptions

### Logistic Regression
- **Type**: Linear model
- **Best for**: Binary/multiclass with linear decision boundaries
- **Pros**: Fast, interpretable, probabilistic outputs
- **Cons**: Assumes linear relationships

### Decision Tree
- **Type**: Tree-based model
- **Best for**: Non-linear relationships, feature interactions
- **Pros**: Interpretable, handles non-linear data
- **Cons**: Can overfit, unstable

### k-Nearest Neighbors (kNN)
- **Type**: Instance-based learning
- **Best for**: Small to medium datasets with clear clusters
- **Pros**: Simple, no training phase
- **Cons**: Slow prediction, sensitive to scale

### Naive Bayes
- **Type**: Probabilistic classifier
- **Best for**: High-dimensional data, text classification
- **Pros**: Fast, works well with limited data
- **Cons**: Assumes feature independence

### Random Forest
- **Type**: Ensemble of decision trees
- **Best for**: Complex patterns, robust predictions
- **Pros**: Reduces overfitting, handles non-linear data
- **Cons**: Less interpretable, slower

### XGBoost
- **Type**: Gradient boosting
- **Best for**: Competition-grade performance
- **Pros**: High accuracy, handles missing values
- **Cons**: More parameters to tune, slower training

## Dataset Requirements

### Minimum Requirements
- ✅ 500+ rows
- ✅ 12+ features (excluding target)
- ✅ CSV format with headers
- ✅ One target column (binary or multiclass)

### Recommended Format
```csv
feature_1,feature_2,...,feature_n,target
1.2,3.4,...,5.6,ClassA
2.3,4.5,...,6.7,ClassB
...
```

### Supported Data Types
- **Numeric**: Integers, floats (used directly)
- **Categorical**: Strings (automatically encoded)
- **Target**: Numeric or categorical (automatically handled)

## Tips for Best Results

### Data Preparation
1. **Clean your data**: Remove or impute missing values
2. **Remove duplicates**: Ensure data quality
3. **Balance classes**: If possible, avoid extreme imbalance
4. **Feature engineering**: Create meaningful features

### Model Selection
1. **Start with all models**: Compare performance
2. **Consider your needs**: 
   - Need interpretability? → Logistic Regression or Decision Tree
   - Need speed? → Naive Bayes or Logistic Regression
   - Need accuracy? → Random Forest or XGBoost
   - Limited data? → Naive Bayes
3. **Check multiple metrics**: Don't rely on accuracy alone

### Interpreting Results
1. **Look at F1 Score**: Good balance of precision and recall
2. **Check confusion matrix**: Understand error patterns
3. **Compare all metrics**: Different metrics for different needs
4. **Consider context**: Domain-specific requirements

## Troubleshooting

### Error: "Dataset has less than 500 rows"
- Your dataset needs at least 500 rows
- Consider generating more data or using a different dataset

### Error: "Dataset has less than 12 features"
- Your dataset needs at least 12 features (excluding target)
- Consider feature engineering to create more features

### Warning: Categorical columns encoded
- This is normal - the app automatically encodes text columns
- Original values are preserved for display

### Models taking too long to train
- Large datasets may take longer
- Consider using a smaller sample for testing
- Random Forest and XGBoost are slower than other models

### Poor model performance
- Check if features are informative
- Try feature engineering or selection
- Consider data imbalance issues
- Verify data quality (no errors, proper encoding)

## Advanced Usage

### Custom Test Size
Adjust the test size slider to control train/test split:
- **Smaller (10-15%)**: More training data, but less reliable evaluation
- **Larger (30-40%)**: More reliable evaluation, but less training data
- **Default (20%)**: Good balance for most cases

### Random State
Change the random state to:
- Get different train/test splits
- Test model stability across different splits
- Default: 42 (for reproducibility)

### Model Subset
Train only specific models to:
- Save time on large datasets
- Compare specific algorithms
- Focus on models relevant to your use case

## Examples

### Example 1: Using Sample Dataset
```bash
# Generate sample data
python generate_dataset.py

# Run app
streamlit run app.py

# In the app:
# 1. Upload sample_dataset.csv
# 2. Select "target" as target column
# 3. Click "Train Models"
# 4. View results
```

### Example 2: Custom Dataset
```python
# Create your dataset
import pandas as pd
import numpy as np

# Your data
data = {
    'feature1': [1, 2, 3, ...],  # At least 12 features
    'feature2': [4, 5, 6, ...],
    # ... more features
    'target': ['A', 'B', 'A', ...]  # Your target
}

df = pd.DataFrame(data)
df.to_csv('my_dataset.csv', index=False)
```

Then upload `my_dataset.csv` in the app.

## Performance Benchmarks

Using the sample dataset (1000 rows, 17 features):

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 0.8450 | 0.8436 | ~1 second |
| Decision Tree | 0.8450 | 0.8426 | ~1 second |
| k-Nearest Neighbors | 0.8850 | 0.8836 | ~1 second |
| Naive Bayes | 0.8200 | 0.8196 | <1 second |
| Random Forest | 0.9000 | 0.8986 | ~2 seconds |
| XGBoost | 0.9400 | 0.9397 | ~3 seconds |

**Best Model**: XGBoost (F1: 0.9397)

## Support

For issues or questions:
1. Check this guide first
2. Review the README.md
3. Check DEPLOYMENT.md for deployment issues
4. Open an issue on GitHub

## Credits

- Built with Streamlit
- ML models from scikit-learn and XGBoost
- Visualizations with matplotlib and seaborn

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Author**: Premchand Chadalawada
