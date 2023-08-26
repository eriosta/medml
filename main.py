import streamlit as st
import streamlit_toggle as tog
from data import setup_kaggle_env_variables, download_kaggle_dataset
from eda import generate_eda
import pandas as pd
import os
from models import prepare_data, train_and_evaluate_models, plot_evaluation_metrics, get_model_hyperparameters, LogisticRegression, RandomForestClassifier, XGBClassifier, DecisionTreeClassifier

st.title("MEDML")

# Initialize dataset and dataset name in session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.dataset_name = None  # Add a state to store the dataset name

# Display the currently loaded dataset information on the sidebar
if st.session_state.dataset_name:
    st.sidebar.markdown(f"📊 **Loaded Dataset:** {st.session_state.dataset_name}")
else:
    st.sidebar.info("No dataset currently loaded")

nav = st.sidebar.radio("Navigation", ["Data", "Exploratory Data Analysis", "Models"])

if nav == "Data":
    data_source = st.sidebar.radio("Choose Data Source", ["Kaggle", "Upload CSV"])

    if data_source == "Kaggle":
        kaggle_username = st.sidebar.text_input("Kaggle Username")
        kaggle_key = st.sidebar.text_input("Kaggle Key", type="password")

        if kaggle_username and kaggle_key:
            try:
                setup_kaggle_env_variables(kaggle_username, kaggle_key)
                st.sidebar.success("Kaggle setup complete!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

        existing_datasets = [
            "jillanisofttech/brain-stroke-dataset",
            "akshaydattatraykhare/diabetes-dataset",
            "fedesoriano/heart-failure-prediction",
            "mathurinache/sepsis-survival-minimal-clinical-records",
            "mirichoi0218/insurance",
            "protobioengineering/mit-bih-arrhythmia-database-modern-2023"
        ]

        dataset_choice = st.sidebar.selectbox("Choose a Kaggle Dataset", ["Other Kaggle Dataset"] + existing_datasets)
        dataset_path = dataset_choice if dataset_choice != "Other Kaggle Dataset" else st.sidebar.text_input("Enter Kaggle Dataset Path")

        if dataset_path:
            try:
                zip_name = download_kaggle_dataset(dataset_path)
                zip_path = os.path.join(zip_name)
                st.sidebar.success(f"{dataset_path} downloaded!")
                st.session_state.dataset_name = dataset_path.split("/")[-1]  # store dataset name to session state

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
            st.session_state.dataset_name = uploaded_file.name  # store actual filename to session state
            st.write(st.session_state.df.head())


elif nav == "Exploratory Data Analysis":
    if st.session_state.df is not None:
        try:
            report_data = generate_eda(st.session_state.df)
            st.sidebar.download_button(
                label="Download EDA Report",
                data=report_data,
                file_name="EDA.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
    else:
        st.warning("Please upload a dataset first under 'Data'.")

elif nav == "Models":
    if st.session_state.df is not None:
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
        test_size = col1.slider('Select Test Set Size (in percentage)', min_value=5, max_value=50, value=20) / 100

        # Display the train set size in the next column
        col2.write(f"Train Set Size: {(1 - test_size) * 100:.0f}%")

        X_train, X_test, y_train, y_test = prepare_data(st.session_state.df, VAR, training_vars, categorical_features, test_size)
        
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
            results = train_and_evaluate_models(X_train, y_train, X_test, y_test, selected_models, optimize_hyperparams)
            st.write(results)

            plot_evaluation_metrics(selected_models, X_test, y_test)
    else:
        st.warning("Please upload a dataset first under 'Dataset Setup'.")
