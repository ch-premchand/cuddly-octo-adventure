# Machine Learning Classification Models Comparison

A comprehensive Streamlit web application for comparing 6 different machine learning classification models on custom datasets.

## 🎯 Features

- **6 Classification Models:**
  - Logistic Regression
  - Decision Tree
  - k-Nearest Neighbors (kNN)
  - Naive Bayes
  - Random Forest
  - XGBoost

- **Comprehensive Metrics:**
  - Accuracy
  - AUC (Area Under the ROC Curve)
  - Precision
  - Recall
  - F1 Score
  - MCC (Matthews Correlation Coefficient)

- **Interactive Web Interface:**
  - CSV file upload
  - Model selection
  - Real-time training
  - Metrics visualization
  - Confusion matrix display
  - Model comparison charts

## 📋 Requirements

- Dataset requirements:
  - Minimum 500 rows
  - Minimum 12 features (plus 1 target column)
  - CSV format with headers
  - Binary or multiclass classification

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ch-premchand/cuddly-octo-adventure.git
cd cuddly-octo-adventure
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Generate Sample Dataset

To create a sample dataset for testing:

```bash
python generate_dataset.py
```

This will create a `sample_dataset.csv` file with:
- 1000 rows
- 17 columns (15 numeric features + 2 categorical features + 1 target)
- Binary classification target

### Using the App

1. **Upload Dataset**: Click "Browse files" in the sidebar and upload your CSV file
2. **Select Target**: Choose which column is your target/label
3. **Configure**: Adjust test size and random state if needed
4. **Select Models**: Choose which models to train (or use all 6)
5. **Train**: Click "Train Models" button
6. **View Results**: Explore metrics, confusion matrices, and comparison charts

## 📊 Example Output

The application displays:
- Dataset overview (rows, columns, features)
- Data preview
- Individual model results with:
  - All 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  - Confusion matrix heatmap
- Summary comparison table with highlighting
- Best model recommendation
- Visual comparison charts for all metrics

## 🛠️ Technical Details

### Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML models and metrics
- **xgboost**: XGBoost classifier
- **matplotlib**: Plotting
- **seaborn**: Statistical visualizations

### Model Implementation

All models use scikit-learn's standard API with default or optimized parameters:
- Train/test split with configurable ratio
- Feature scaling for distance-based models (Logistic Regression, kNN)
- Support for both binary and multiclass classification
- Automatic handling of categorical features

## 📁 Project Structure

```
cuddly-octo-adventure/
├── app.py                  # Main Streamlit application
├── generate_dataset.py     # Sample dataset generator
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── sample_dataset.csv     # Generated sample dataset (after running script)
```

## 🎓 Educational Purpose

This project demonstrates:
- Implementation of multiple ML classification algorithms
- Proper model evaluation with multiple metrics
- Interactive data science applications with Streamlit
- Best practices for ML model comparison
- Data preprocessing and feature engineering

## 📝 License

This project is created for educational purposes.

## 👤 Author

Premchand Chadalawada

## 🔗 Links

- GitHub Repository: https://github.com/ch-premchand/cuddly-octo-adventure
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Documentation: https://scikit-learn.org/