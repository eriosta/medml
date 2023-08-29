import pandas as pd
import streamlit as st
import streamlit_toggle as tog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, 
                            mean_absolute_error, mean_squared_error, classification_report)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import shap
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
    
    hyperparams_config = {
        "Logistic Regression": {
            "header": "Logistic Regression Hyperparameters",
            "params": {
                "C": {
                    "type": "slider",
                    "desc": "Regularization strength (C): Inverse of regularization strength. Smaller values specify stronger regularization.",
                    "args": [0.001, 100.0, 1.0, 0.001]
                },
                "max_iter": {
                    "type": "slider",
                    "desc": "Maximum Iterations: Maximum number of iterations for the solver to converge.",
                    "args": [1000, 20000, 10000, 1000]
                },
                "class_weight": {
                    "type": "selectbox",
                    "desc": "Class Weight: Weights associated with classes. Useful for imbalanced datasets.",
                    "args": [None, "balanced"]
                }
            }
        },
        "Random Forest Classifier": {
            "header": "Random Forest Classifier Hyperparameters",
            "params": {
                "n_estimators": {
                    "type": "slider",
                    "desc": "Number of Trees (n_estimators): The number of trees in the forest.",
                    "args": [50, 150, 100, 10]
                },
                "max_depth": {
                    "type": "selectbox",
                    "desc": "Max Depth: The maximum depth of the tree.",
                    "args": [None, 5, 10]
                }
            }
        },
        "Gradient Boosting Classifier": {
            "header": "Gradient Boosting Classifier Hyperparameters",
            "params": {
                "learning_rate": {
                    "type": "slider",
                    "desc": "Learning Rate: Shrinks the contribution of each tree. There's a trade-off between learning rate and number of boosting rounds.",
                    "args": [0.01, 0.1, 0.05, 0.01]
                },
                "n_estimators": {
                    "type": "slider",
                    "desc": "Number of Boosting Rounds (n_estimators): The number of boosting stages to run.",
                    "args": [50, 150, 100, 10]
                }
            }
        },
        "Decision Tree Classifier": {
            "header": "Decision Tree Classifier Hyperparameters",
            "params": {
                "max_depth": {
                    "type": "selectbox",
                    "desc": "Max Depth: The maximum depth of the tree.",
                    "args": [None, 5, 10]
                }
            }
        }
    }

    params = {}

    if model_name in hyperparams_config:
        st.subheader(hyperparams_config[model_name]["header"])

        for param, config in hyperparams_config[model_name]["params"].items():
            st.write(config["desc"])
            
            if config["type"] == "slider":
                params[param] = st.slider(param, *config["args"])
            elif config["type"] == "selectbox":
                params[param] = st.selectbox(param, config["args"], key=f"{model_name}_{param}")

    return params

def plot_evaluation_metrics(models, X_test, y_test, VAR):
    """Plot confusion matrices and ROC-AUC curves."""
    
    def plot_roc(ax, y_test, y_prob, method):
        """Helper function to plot the ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob), estimator_name=method).plot(ax=ax)
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_title(f'ROC Curve: {method}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')

    num_models = len(models)
    fig, axs = plt.subplots(num_models, 2, figsize=(12, 5 * num_models))
    if num_models == 1:
        axs = np.array([axs])

    for idx, (method, model) in enumerate(models.items()):
        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[f'No {VAR}', VAR]).plot(ax=axs[idx, 0], cmap='Blues', values_format=".2f")

        # ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc(axs[idx, 1], y_test, y_prob, method)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models):
    """Train and evaluate machine learning models."""
    results = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
    progress = st.progress(0)

    for idx, (method, model) in enumerate(models.items()):
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Update the results dataframe
        results = results.append({
            'Method': method,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }, ignore_index=True)
        
        # Update progress bar
        progress.progress((idx + 1) / len(models))

    return results, models

def perform_shap(models, selected_model_name, X_train, X_test):
    # Compute SHAP values for the selected model
    selected_model = models[selected_model_name]
    explainer = shap.Explainer(selected_model, X_train)
    shap_values = explainer(X_test)

    # Display SHAP summary bar plot
    st.write("SHAP Summary Bar Plot:")
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    st.pyplot(plt.gcf())
    plt.close()

    # Display SHAP summary violin plot
    st.write("SHAP Summary Violin Plot:")
    shap.summary_plot(shap_values, X_test, plot_type='violin', show=False)
    st.pyplot(plt.gcf())
    plt.close()

    # Scatter all interactions if user wants
    available_columns = list(X_test.columns)
    for feature in available_columns:
        st.write(f"Scatter Plot for {feature}:")
        shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

def display_data_option():
    """Option to display a head of the data."""
    if st.checkbox("View data? (Head Only)"):
        st.dataframe(st.session_state.df.head())
    
def select_training_parameters():
    """Return target variable, training variables, and categorical features."""
    VAR = st.selectbox("Select Target Variable", st.session_state.df.columns)
    training_vars = st.multiselect(
        "Select Variables for Training (excluding target)", 
        st.session_state.df.columns.difference([VAR])
    )
    categorical_features = st.multiselect(
        "Select Categorical Variables to Encode", training_vars
    )
    return VAR, training_vars, categorical_features

def choose_train_test_sizes():
    """Return train and test set sizes."""
    col1, col2 = st.columns(2)
    test_size = col1.slider('Select Test Set Size (%)', min_value=5, max_value=50, value=20) / 100
    col2.write(f"Train Set Size: {(1 - test_size) * 100:.0f}%")
    return test_size

from sklearn.utils.class_weight import compute_class_weight

def train():
  
  if 'df' in st.session_state:
    
    display_data_option()
    
    VAR, training_vars, categorical_features = select_training_parameters()
    test_size = choose_train_test_sizes()
    
    X_train, X_test, y_train, y_test = prepare_data(
        st.session_state.df, VAR, training_vars, categorical_features, test_size
    )
    st.session_state['data_split'] = (X_train, X_test, y_train, y_test)
    
    model_selection = st.multiselect(
        "Select Models", 
        ["Logistic Regression", "Gradient Boosting Classifier"]
    )
    
    balance_class_weights = st.checkbox("Balance Class Weights?")
    
    if balance_class_weights:
        neg, pos = np.bincount(y_train)
        scale_pos_weight_value = neg / pos
    else:
        scale_pos_weight_value = 1
    
    optimize_hyperparams = st.checkbox("Optimize Hyperparameters?")
    
    selected_models = {}
    all_models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced' if balance_class_weights else None),
        'Random Forest Classifier': RandomForestClassifier(class_weight='balanced' if balance_class_weights else None),
        'Gradient Boosting Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value),
        'Decision Tree Classifier': DecisionTreeClassifier(class_weight='balanced' if balance_class_weights else None)
    }
    
    for model in model_selection:
        base_model = all_models[model]
        if optimize_hyperparams:
            user_defined_params = get_model_hyperparameters(model)  # Get user-defined hyperparameters
            base_model.set_params(**user_defined_params)           # Update base model with those hyperparameters
        selected_models[model] = base_model
    
    if st.button("Train Models"):
        results, trained_models = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, selected_models)
    
        st.session_state.trained_models = trained_models  # Saving the models to session state
        st.session_state.results = results
    
        st.write(results)
    
        plot_evaluation_metrics(selected_models, X_test, y_test, VAR)
    
        # Compute AUC for each model and find the best one
        results['AUC'] = [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for model in trained_models.values()]
        best_model_name = results['Method'][results['AUC'].idxmax()]
        
        # User model selection for SHAP
        selected_model_for_shap = st.selectbox(
            "Choose a model for SHAP analysis:",
            options=list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name)  # Default to the best model
        )
        
        perform_shap(trained_models, selected_model_for_shap, X_train, X_test)
  else:
    st.warning("Please load data source under **Data** first")
