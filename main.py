import streamlit as st
import streamlit_toggle as tog
from data import data_run, transform
from eda import generate_eda
import pandas as pd
import os
from models import train, perform_shap, prepare_data, train_and_evaluate_models, plot_evaluation_metrics, get_model_hyperparameters, LogisticRegression, RandomForestClassifier, XGBClassifier, DecisionTreeClassifier
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

nav = st.sidebar.radio("Start Building", ["Get Started","Data", "Models", "Extra"])

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
    # Add a sub-page for model explanation in the sidebar using radio buttons
    data_page = st.sidebar.radio("Navigate", ['Source', 'Exploratory Analysis','Transformation'])

    if data_page == 'Source':
            data_run()

    elif data_page == "Exploratory Analysis":
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
            st.warning("Please upload a dataset under **Data** first.")
    elif data_page == "Transformation":
         transform()
        
elif nav == "Models":

    if st.session_state.df is not None:
        # Add a sub-page for model explanation in the sidebar using radio button
        model_page = st.sidebar.radio("Navigate", ['Train & Evaluate', 'Explain'])
     
        if model_page == 'Train & Evaluate':
            train()
            
        if model_page == 'Explain':

            if 'trained_models' in st.session_state:

                trained_models = st.session_state.trained_models
                X_train, X_test, y_train, y_test = st.session_state['data_split']
                results = st.session_state['results']
               
                perform_shap(trained_models,
                                X_train, X_test, y_test,
                                results)
            else:
                st.warning("Please train a model under Train & Evaluate first.")

    else:
        st.warning("Please upload a dataset under **Data** first.")

if nav == "Extra":
    nav2 = st.sidebar.radio("Extra", ["Learn","Generative AI"])

    if nav2 == "Learn":
        show()

    if nav2 == "Generative AI":
        # Sidebar Navigation
        navigation = st.sidebar.radio('Navigation', ['Llama2'])

        if navigation == 'Llama2':
            llama2()
