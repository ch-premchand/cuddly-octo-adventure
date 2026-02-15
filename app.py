import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="ML Classification Models", layout="wide")

st.title("🤖 Machine Learning Classification Models Comparison")
st.markdown("Upload your dataset and compare 6 different classification models!")

# Sidebar for file upload and model selection
st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Features", df.shape[1] - 1)
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Dataset validation
    if df.shape[0] < 500:
        st.warning(f"⚠️ Dataset has {df.shape[0]} rows. Recommended minimum is 500 rows.")
    if df.shape[1] < 13:
        st.warning(f"⚠️ Dataset has {df.shape[1]} columns (including target). Recommended minimum is 13 (12 features + 1 target).")
    
    # Select target column
    st.sidebar.subheader("Target Selection")
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    
    if target_column:
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle non-numeric features
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            st.info(f"Encoding categorical columns: {', '.join(categorical_columns)}")
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            st.info(f"Target classes: {', '.join(map(str, le_target.classes_))}")
        
        # Check if binary or multiclass
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2
        
        st.sidebar.subheader("Model Configuration")
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        models_to_run = st.sidebar.multiselect(
            "Select Models to Compare",
            ["Logistic Regression", "Decision Tree", "k-Nearest Neighbors", 
             "Naive Bayes", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Decision Tree", "k-Nearest Neighbors", 
                     "Naive Bayes", "Random Forest", "XGBoost"]
        )
        
        if st.sidebar.button("Train Models", type="primary"):
            results = []
            
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                "k-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
                "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss')
            }
            
            st.header("Model Training Results")
            
            for model_name in models_to_run:
                if model_name in models:
                    with st.expander(f"📊 {model_name}", expanded=True):
                        model = models[model_name]
                        
                        # Train model
                        with st.spinner(f"Training {model_name}..."):
                            if model_name in ["Logistic Regression", "k-Nearest Neighbors"]:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                y_pred_proba = model.predict_proba(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        mcc = matthews_corrcoef(y_test, y_pred)
                        
                        # Calculate AUC
                        if is_binary:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            st.metric("Precision", f"{precision:.4f}")
                        with col2:
                            st.metric("Recall", f"{recall:.4f}")
                            st.metric("F1 Score", f"{f1:.4f}")
                        with col3:
                            st.metric("AUC", f"{auc:.4f}")
                            st.metric("MCC", f"{mcc:.4f}")
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {model_name}')
                        st.pyplot(fig)
                        plt.close()
                        
                        # Store results
                        results.append({
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'AUC': auc,
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1,
                            'MCC': mcc
                        })
            
            # Summary comparison
            if results:
                st.header("📈 Model Comparison Summary")
                results_df = pd.DataFrame(results)
                
                st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']))
                
                # Best model
                best_model_idx = results_df['F1 Score'].idxmax()
                best_model = results_df.loc[best_model_idx, 'Model']
                st.success(f"🏆 Best Model (by F1 Score): **{best_model}**")
                
                # Visualization
                st.subheader("Metrics Comparison")
                metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                for idx, metric in enumerate(metrics_to_plot):
                    axes[idx].bar(results_df['Model'], results_df[metric], color='skyblue')
                    axes[idx].set_title(metric)
                    axes[idx].set_ylabel('Score')
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    st.markdown("""
    ### Requirements:
    - Dataset should have **at least 500 rows**
    - Dataset should have **at least 12 features** (plus 1 target column)
    - CSV format with headers
    - One column should be the target/label
    
    ### Supported Models:
    1. **Logistic Regression** - Linear model for classification
    2. **Decision Tree** - Tree-based model
    3. **k-Nearest Neighbors (kNN)** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting framework
    
    ### Computed Metrics:
    - **Accuracy** - Overall correctness
    - **AUC** - Area Under the ROC Curve
    - **Precision** - Positive predictive value
    - **Recall** - Sensitivity/True positive rate
    - **F1 Score** - Harmonic mean of precision and recall
    - **MCC** - Matthews Correlation Coefficient
    """)
