import streamlit as st
import streamlit_toggle as tog
from data import data_run
from eda import generate_eda
import pandas as pd
import os
from models import prepare_data, train_and_evaluate_models, plot_evaluation_metrics, get_model_hyperparameters, LogisticRegression, RandomForestClassifier, XGBClassifier, DecisionTreeClassifier, run_shap
from learn import show
from chat import llama2

st.sidebar.title("MEDML")

# Initialize dataset and dataset name in session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.dataset_name = None  # Add a state to store the dataset name

# Display the currently loaded dataset information on the sidebar
if st.session_state.dataset_name:
    st.sidebar.markdown(f"📊 **Loaded Dataset:** {st.session_state.dataset_name}")
else:
    st.sidebar.info("No dataset currently loaded")

nav = st.sidebar.radio("Start Building", ["Get Started","Data", "Exploratory Data Analysis", "Models", "Extra"])

if nav == "Get Started":
    st.markdown("""
    # Welcome to MEDML!
    ### A Machine Learning Primer Built By Physicians For Physicians.

    MEDML makes machine learning in medicine accessible to all.

    ### 1. Data Source 📊

    **Kaggle:** 
    - Community-driven datasets platform.
    - New? Sign up on [Kaggle](https://www.kaggle.com/).
    - For MEDML access: Profile (top right) -> 'Account' -> `API` -> `Create New API Token`. Use "username" and "key" from downloaded file here.
    
    **Upload CSV:** 
    - Ensure your data is in CSV format.
    - Use 'Upload CSV' to input.

    ### 2. Explore Trends  🔍

    After uploading, check 'Exploratory Data Analysis' for:
    - Dataset overview.
    - Visualization insights.

    ### 3. Model, Predict and Evaluate 🧠

    - **Target**: Define the outcome.
    - **Variables**: List influencing factors.
    - **Predictor**: Pick one or more; MEDML handles the details.
    - **Optimize Hyperparameters**: Enhance prediction accuracy.

    ---

    MEDML aids, but doesn't replace, medical judgment. Need help? [Contact us]().
    """)

    st.markdown("Use the sidebar to start with MEDML!")

elif nav == "Data":
    data_run()

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
        st.warning("Please upload a dataset first under Data.")

elif nav == "Models":
    if st.session_state.df is not None:
        
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
            results, trained_models = train_and_evaluate_models(X_train, y_train, X_test, y_test, selected_models, optimize_hyperparams)
            st.write(results)
            plot_evaluation_metrics(selected_models, X_test, y_test, VAR)
        
    else:
        st.warning("Please upload a dataset first under Data.")

if nav == "Extra":
    nav2 = st.sidebar.radio("Extra", ["Learn","Generative AI"])

    if nav2 == "Learn":
        show()

    if nav2 == "Generative AI":
        # Sidebar Navigation
        navigation = st.sidebar.radio('Navigation', ['Llama2'])

        if navigation == 'Llama2':
            llama2()
