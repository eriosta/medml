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
# import pickle

def prepare_data(df, target_var, training_vars, categorical_vars, test_size=0.2):
    # Extract features and target from dataframe
    X = df[training_vars]
    y = df[target_var]

    # Encode categorical variables (if there are any)
    if categorical_vars:
        X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def get_model_hyperparameters(model_name):
    """Retrieve user-defined hyperparameters for a given model."""
    params = {}

    if model_name == "Logistic Regression":
        st.subheader("Logistic Regression Hyperparameters")
        
        st.write("Regularization strength (C): Inverse of regularization strength. Smaller values specify stronger regularization.")
        params["C"] = st.slider("C", 0.001, 100.0, 1.0, 0.001)
        
        st.write("Maximum Iterations: Maximum number of iterations for the solver to converge.")
        params["max_iter"] = st.slider("Max Iterations", 1000, 20000, 10000, 1000)
        
        st.write("Class Weight: Weights associated with classes. Useful for imbalanced datasets.")
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"], key=f"{model_name}_class_weight")
        
    elif model_name == "Random Forest Classifier":
        st.subheader("Random Forest Classifier Hyperparameters")
        
        st.write("Number of Trees (n_estimators): The number of trees in the forest.")
        params["n_estimators"] = st.slider("Number of Trees (n_estimators)", 50, 150, 100, 10)
        
        st.write("Max Depth: The maximum depth of the tree.")
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10], key=f"{model_name}_max_depth")

    elif model_name == "Gradient Boosting Classifier":
        st.subheader("Gradient Boosting Classifier Hyperparameters")
        
        st.write("Learning Rate: Shrinks the contribution of each tree. There's a trade-off between learning rate and number of boosting rounds.")
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.1, 0.05, 0.01)
        
        st.write("Number of Boosting Rounds (n_estimators): The number of boosting stages to run.")
        params["n_estimators"] = st.slider("Number of Boosting Rounds (n_estimators)", 50, 150, 100, 10)

    elif model_name == "Decision Tree Classifier":
        st.subheader("Decision Tree Classifier Hyperparameters")
        
        st.write("Max Depth: The maximum depth of the tree.")
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10], key=f"{model_name}_max_depth")

    return params

def plot_evaluation_metrics(models, X_test, y_test, VAR):
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
        
        ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[f'No {VAR}', VAR]).plot(ax=axs[idx, 0], cmap='Blues', values_format=".2f")

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

    return results, models

def perform_shap(models, X_train, X_test, y_test, results):
    """Perform SHAP analysis on the best model."""
    # Compute AUC and add it to the results DataFrame
    results['AUC'] = [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for model in models.values()]

    best_method = results.loc[results['AUC'].idxmax()]['Method']
    st.write(f"Best performing method (based on highest AUC): {best_method}")

    # Perform SHAP explanations if user agrees
    best_model = models[best_method]
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)

    # SHAP summary plot
    st.write("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test)

    # Allow user to select which interactions to display
    available_columns = list(X_test.columns)
    # VARIABLESELECTED = st.selectbox("Choose feature for dependence plot", available_columns)
    # st.write("SHAP Dependence Plot:")
    # shap.dependence_plot(VARIABLESELECTED, shap_values, X_test)

    # Scatter all interactions if user wants
    if st.checkbox('Show scatter plots for all features using SHAP values?'):
        for feature in available_columns:
            st.write(f"Scatter Plot for {feature}:")
            shap.plots.scatter(shap_values[:, feature], color=shap_values)

    # Display SHAP force plot for specific observation if user agrees
    if st.checkbox('Display SHAP force plot for a specific observation?'):
        index = st.number_input('Enter observation index:', min_value=0, max_value=len(X_test)-1, step=1)
        shap.plots.force(explainer.expected_value[0], shap_values[index], X_test.iloc[index])
