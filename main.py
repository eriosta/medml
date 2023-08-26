import streamlit as st
from data import setup_kaggle_env_variables, download_kaggle_dataset
from eda import generate_eda
import pandas as pd
import os
from models import prepare_data, train_and_evaluate_models, plot_evaluation_metrics, LogisticRegression, RandomForestClassifier, XGBClassifier, DecisionTreeClassifier

st.title("Kaggle Dataset Streamlit App")

if 'df' not in st.session_state:
    st.session_state.df = None

nav = st.sidebar.radio("Navigation", ["Dataset Setup", "EDA Report", "Model Training"])

if nav == "Dataset Setup":
    data_source = st.sidebar.radio("Choose your data source", ["Kaggle Dataset", "Upload your CSV"])

    if data_source == "Kaggle Dataset":
        kaggle_username = st.sidebar.text_input("Kaggle Username")
        kaggle_key = st.sidebar.text_input("Kaggle Key", type="password")

        if kaggle_username and kaggle_key:
            try:
                setup_kaggle_env_variables(kaggle_username, kaggle_key)
                st.sidebar.success("Kaggle setup complete!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

        existing_datasets = ["jillanisofttech/brain-stroke-dataset"]
        dataset_choice = st.sidebar.selectbox("Choose a Kaggle Dataset", ["Custom Dataset"] + existing_datasets)
        dataset_path = dataset_choice if dataset_choice != "Custom Dataset" else st.sidebar.text_input("Enter Kaggle Dataset Path")

        if dataset_path:
            try:
                zip_name = download_kaggle_dataset(dataset_path)
                zip_path = os.path.join(zip_name)
                st.sidebar.success(f"{dataset_path} downloaded!")

                with open(zip_path, 'rb') as f:
                    bytes_data = f.read()
                st.sidebar.download_button(
                    label="Download Dataset Zip",
                    data=bytes_data,
                    file_name=f"{dataset_choice}.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file:
            st.sidebar.success("CSV file uploaded!")
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write(st.session_state.df.head())

elif nav == "EDA Report":
    if st.session_state.df is not None:
        try:
            report_data = generate_eda(st.session_state.df)
            st.sidebar.download_button(
                label="Download EDA Report",
                data=report_data,
                file_name="eda_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
    else:
        st.warning("Please upload a dataset first under 'Dataset Setup'.")

elif nav == "Model Training":
    if st.session_state.df is not None:
        VAR = st.selectbox("Select Target Variable", st.session_state.df.columns)
        categorical_features = st.multiselect("Select Categorical Variables", st.session_state.df.columns)
        
        X_train, X_test, y_train, y_test = prepare_data(st.session_state.df, VAR, categorical_features)
        
        model_selection = st.multiselect("Select Models", ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier", "Decision Tree Classifier"])
        
        # Choice for Hyperparameter Optimization
        optimize_hyperparams = st.checkbox('Optimize Hyperparameters?')

        selected_models = {}
        all_models = {
            'Logistic Regression': LogisticRegression(max_iter=10000, class_weight='balanced'),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'Decision Tree Classifier': DecisionTreeClassifier()
        }

        for model in model_selection:
            selected_models[model] = all_models[model]

        if st.button("Train Models"):
            results = train_and_evaluate_models(X_train, y_train, X_test, y_test, selected_models, optimize_hyperparams)
            st.write(results)

            plot_evaluation_metrics(selected_models, X_test, y_test)
    else:
        st.warning("Please upload a dataset first under 'Dataset Setup'.")
