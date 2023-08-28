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

def plot_evaluation_metrics(models, X_test, y_test, VAR, task_type):
    """Plot evaluation metrics based on the task type."""
    num_models = len(models)

    if task_type == "regression":
        # For regression, there's no need for confusion matrix and ROC curve
        st.write("For regression tasks, use appropriate metrics like MAE, MSE, etc.")
        return

    if num_models == 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = np.array([axs])
    else:
        fig, axs = plt.subplots(num_models, 2, figsize=(12, 5 * num_models))

    for idx, (method, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)

        if task_type == "binary_classification":
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'No {VAR}', VAR]).plot(ax=axs[idx, 0])
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob), estimator_name=method)
            roc_disp.plot(ax=axs[idx, 1])
        elif task_type == "multiclass_classification":
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axs[idx, 0])
            # One-vs-all ROC curves can be plotted if needed

        axs[idx, 1].set_title(f'Evaluation Metric: {method}')
        axs[idx, 1].set_xlim([0.0, 1.0])
        axs[idx, 1].set_ylim([0.0, 1.05])
        axs[idx, 1].set_ylabel('True Positive Rate')
        axs[idx, 1].set_xlabel('False Positive Rate')

    plt.tight_layout()
    st.pyplot(fig)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, task_type, optimize_hyperparams=False):
    """Train and evaluate machine learning models."""
    
    # Ensure proper task_type
    assert task_type in ["binary_classification", "multiclass_classification", "regression"], "Invalid task type specified"

    results_list = []

    # Parameter grid for hyperparameter optimization.
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
        try:
            # Hyperparameter optimization.
            if optimize_hyperparams and method in param_grids:
                st.write(f"Optimizing hyperparameters for {method}...")
                grid_search = GridSearchCV(model, param_grids[method], cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            # Store results depending on task type.
            metrics = {}
            if task_type in ["binary_classification", "multiclass_classification"]:
                avg_method = 'weighted' if task_type == "multiclass_classification" else 'binary'
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average=avg_method),
                    'Recall': recall_score(y_test, y_pred, average=avg_method),
                    'F1': f1_score(y_test, y_pred, average=avg_method)
                }
            elif task_type == "regression":
                metrics = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred)
                }

            results_list.append({'Method': method, **metrics})
            
            # Update trained model
            models[method] = best_model

            # Update progress.
            model_count += 1
            progress.progress(model_count / total_models)

        except Exception as e:
            st.warning(f"An error occurred while processing {method}: {str(e)}")

    results = pd.DataFrame(results_list)
    return results, models

def perform_shap(models, X_train, X_test, y_test, results):
    """Perform SHAP analysis on the best model."""
    
    # Compute AUC and add it to the results DataFrame
    results['AUC'] = [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for model in models.values()]

    best_method = results.loc[results['AUC'].idxmax()]['Method']
    best_auc = results.loc[results['AUC'].idxmax()]['AUC']  # Extract the AUC of the best model
    
    # Display the best method and its AUC
    st.write(f"Best performing method (based on highest AUC): {best_method}")
    st.write(f"AUC for {best_method}: {best_auc:.4f}")  # Displaying the AUC with 4 decimal points
    
    # Perform SHAP explanations if user agrees
    best_model = models[best_method]
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)

    # SHAP summary plot
    st.write("SHAP Summary Bar Plot:")
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    st.write("SHAP Summary Violin Plot:")
    shap.summary_plot(shap_values, X_test, plot_type='violin', show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # Scatter all interactions if user wants
    available_columns = list(X_test.columns)
    if st.checkbox('Show interaction plots for all features?'):
        for feature in available_columns:
            st.write(f"Scatter Plot for {feature}:")
            shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

def train():
    # Create a checkbox to decide whether to display the data head or not
    if st.checkbox("View data? (Head Only)"):
        st.dataframe(st.session_state.df.head())

    # Select the target variable
    VAR = st.selectbox("Select Target Variable", st.session_state.df.columns)

    # Select the training variables
    training_vars = st.multiselect("Select Variables for Training (excluding target)", 
                                st.session_state.df.columns.difference([VAR]))

    # From the selected training variables, select the categorical ones for encoding
    categorical_features = st.multiselect("Select Categorical Variables to Encode", training_vars)

    # Create two columns to display train and test percentages side by side
    col1, col2 = st.columns(2)

    # Slider for test set size
    test_size = col1.slider('Select Test Set Size (%)', min_value=5, max_value=50, value=20) / 100

    # Display the train set size in the next column
    col2.write(f"Train Set Size: {(1 - test_size) * 100:.0f}%")

    X_train, X_test, y_train, y_test = prepare_data(st.session_state.df, VAR, training_vars, categorical_features, test_size)

    st.session_state['data_split'] = (X_train, X_test, y_train, y_test)

    model_selection = st.multiselect("Select Models", ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier", "Decision Tree Classifier"])
    
    # Choice for Hyperparameter Optimization using toggle
    optimize_hyperparams = tog.st_toggle_switch(label="Optimize Hyperparameters?", 
                                                key="optimize_hyperparams_key", 
                                                default_value=False, 
                                                label_after=False, 
                                                inactive_color='#D3D3D3', 
                                                active_color="#11567f", 
                                                track_color="#29B5E8")

    selected_models = {}
    all_models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, class_weight='balanced'),
        'Random Forest Classifier': RandomForestClassifier(),
        'Gradient Boosting Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Decision Tree Classifier': DecisionTreeClassifier()
    }

    for model in model_selection:
        selected_models[model] = all_models[model]

        if optimize_hyperparams:
            user_defined_params = get_model_hyperparameters(model)
            base_model = all_models[model]
            # Update base model with user-defined hyperparameters
            base_model.set_params(**user_defined_params)
            selected_models[model] = base_model

    if st.button("Train Models"):
        results, trained_models = train_and_evaluate_models(
            X_train, y_train, X_test, y_test,
            selected_models, 
            optimize_hyperparams)

        st.session_state.trained_models = trained_models  # Saving the models to session state
        st.session_state.results = results

        st.write(results)

        plot_evaluation_metrics(selected_models, X_test, y_test, VAR)