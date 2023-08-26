import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.preprocessing import OneHotEncoder
import shap
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier  # Import the classifier
from time import sleep

def prepare_data(df, VAR, categorical_features):
    """Prepare the dataset for modeling."""
    X = df.drop(VAR, axis=1)  # Features
    y = df[VAR]  # Target

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_model_hyperparameters(model_name):
    """Retrieve user-defined hyperparameters for a given model."""
    params = {}

    if model_name == "Logistic Regression":
        st.subheader("Logistic Regression Hyperparameters")
        params["C"] = st.slider("Regularization strength (C)", 0.001, 100.0, 1.0, 0.001)
        params["max_iter"] = st.slider("Maximum Iterations", 1000, 20000, 10000, 1000)
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"])
        
    elif model_name == "Random Forest Classifier":
        st.subheader("Random Forest Classifier Hyperparameters")
        params["n_estimators"] = st.slider("Number of Trees (n_estimators)", 50, 150, 100, 10)
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10])

    elif model_name == "Gradient Boosting Classifier":
        st.subheader("Gradient Boosting Classifier Hyperparameters")
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.1, 0.05, 0.01)
        params["n_estimators"] = st.slider("Number of Boosting Rounds (n_estimators)", 50, 150, 100, 10)

    elif model_name == "Decision Tree Classifier":
        st.subheader("Decision Tree Classifier Hyperparameters")
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10])

    return params

from time import sleep

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, optimize_hyperparams=False):
    """Train and evaluate machine learning models."""
    results = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])

    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [10000],
            'class_weight': ['balanced']
        },
        'Random Forest Classifier': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10]
        },
        'Gradient Boosting Classifier': {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150]
        },
        'Decision Tree Classifier': {
            'max_depth': [None, 5, 10]
        }
    }

    total_models = len(models)
    progress = st.progress(0)
    model_count = 0

    for method, model in models.items():
        if optimize_hyperparams and method in param_grids:
            st.write(f"Optimizing hyperparameters for {method}...")
            grid_search = GridSearchCV(model, param_grids[method], cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Update the models dict with the trained model
        models[method] = best_model

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        new_row = {'Method': method, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        results.loc[len(results)] = new_row

        model_count += 1
        progress.progress(model_count / total_models)
        sleep(0.1)

    return results

def plot_evaluation_metrics(models, X_test, y_test):
    """Plot confusion matrices and ROC-AUC curves."""
    num_models = len(models)
    
    if num_models == 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = np.array([axs])
    else:
        fig, axs = plt.subplots(num_models, 2, figsize=(12, 5 * num_models))
    
    for idx, (method, model) in enumerate(models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['No Stroke', 'Stroke']).plot(ax=axs[idx, 0], cmap='Blues', values_format=".2f")

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob), estimator_name=method)
        roc_disp.plot(ax=axs[idx, 1])

        axs[idx, 1].set_title(f'ROC Curve: {method}')
        axs[idx, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[idx, 1].set_xlim([0.0, 1.0])
        axs[idx, 1].set_ylim([0.0, 1.05])
        axs[idx, 1].set_ylabel('True Positive Rate')
        axs[idx, 1].set_xlabel('False Positive Rate')

    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this line


