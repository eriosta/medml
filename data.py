import os
import subprocess
import pandas as pd
import streamlit as st
import base64
import numpy as np
from sklearn.impute import KNNImputer

def setup_kaggle_env_variables(username, key):
    """Set up environment variables for Kaggle."""
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

def download_kaggle_dataset(dataset_path):
    """Download the specified kaggle dataset using Kaggle CLI."""
    download_command = f"kaggle datasets download {dataset_path}"
    download_result = subprocess.run(download_command, shell=True, capture_output=True, text=True)
    
    if download_result.returncode != 0:
        raise ValueError(download_result.stderr)
    dataset_name = dataset_path.split("/")[-1] + ".zip"
    return dataset_name

def display_metadata(existing_datasets):
    # Sample dataset descriptions
    dataset_descriptions = [
        "Data related to various factors influencing brain strokes.",
        "Comprehensive set of data related to diabetes patients.",
        "Clinical records for heart failure prediction.",
        "Minimal clinical records related to sepsis survival.",
        "Dataset about insurance claims and details.",
        "Updated arrhythmia dataset from MIT."
    ]

    # Sample tasks for each dataset
    dataset_tasks = [
        "Classification",
        "Classification",
        "Classification",
        "Classification",
        "Regression",
        "Classification"
    ]

    # Extract the dataset name from the full path
    dataset_names = [dataset.split('/')[-1] for dataset in existing_datasets]

    # Create a dataframe
    df_datasets = pd.DataFrame({
        "Path": existing_datasets,
        "Available Datasets": dataset_names,
        "Description": dataset_descriptions,
        "Task": dataset_tasks
    })

    # Display the header
    st.header("Kaggle Datasets")
    
    # Display each dataset in a collapsible section
    for index, row in df_datasets.iterrows():
        with st.expander(row["Available Datasets"]):
            st.write("**Path:**", row["Path"])
            st.write("**Description:**", row["Description"])
            st.write("**Task:**", row["Task"])

def data_run():

    data_source = st.sidebar.radio("Choose Data Source", ["Use demo","Upload file","Kaggle"])

    if data_source == "Use demo":
        st.session_state.df = pd.read_csv("heart_disease_uci.csv")
        st.session_state.dataset_name = "heart_disease_uci"
        st.write(st.session_state.df)

        with st.expander("Metadata: Heart Disease UCI"):
            
            st.markdown("**Patient Demographics**")
            st.markdown("- **id**: Unique id for each patient")
            st.markdown("- **age**: Age of the patient in years")
            st.markdown("- **origin**: Place of study")
            st.markdown("- **sex**: Gender (Male/Female)")
            st.markdown(" ")
            st.markdown("**Clinical Metrics**")
            st.markdown("- **trestbps**: Resting blood pressure (in mm Hg on admission)")
            st.markdown("- **chol**: Serum cholesterol level (in mg/dl)")
            st.markdown("- **fbs**: Fasting blood sugar (> 120 mg/dl indicates True)")
            st.markdown("- **thalach**: Maximum heart rate achieved during a stress test")
            st.markdown("- **oldpeak**: ST depression induced by exercise relative to rest")
            st.markdown("- **cp**: Type of chest pain experienced ([typical angina, atypical angina, non-anginal, asymptomatic])")
            st.markdown("- **restecg**: Resting electrocardiographic results ([normal, stt abnormality, lv hypertrophy])")
            st.markdown("- **exang**: Whether exercise-induced angina was experienced (True/False)")
            st.markdown("- **slope**: Slope of the peak exercise ST segment")
            st.markdown("- **ca**: Number of major blood vessels (0-3) visible via fluoroscopy")
            st.markdown("- **thal**: Heart defect type ([normal, fixed defect, reversible defect])")
            st.markdown(" ")
            st.markdown("**Prediction Outcome**")
            st.markdown("- **num**: Predicted heart disease stage; 0 = no disease; 1-4 indicate increasing severity.")
            
    elif data_source == "Kaggle":

        st.info("""
    ## How to get your Kaggle credentials
    1. Log in to [Kaggle](https://www.kaggle.com/).
    2. Go to your Account page (click on your profile picture on the top-right).
    3. Scroll down to the `API` section.
    4. Click on “Create New API Token”.
    5. This will download a file named `kaggle.json`.
    6. Open the file and you'll find your `username` and `key`.
    """)
            
        kaggle_username = st.sidebar.text_input("Kaggle Username")
        kaggle_key = st.sidebar.text_input("Kaggle Key", type="password")

        if kaggle_username and kaggle_key:
            try:
                setup_kaggle_env_variables(kaggle_username, kaggle_key)
                st.sidebar.success("Kaggle setup complete!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

        existing_datasets = [
            "redwankarimsony/heart-disease-data",
            "jillanisofttech/brain-stroke-dataset",
            "akshaydattatraykhare/diabetes-dataset",
            "fedesoriano/heart-failure-prediction",
            "mathurinache/sepsis-survival-minimal-clinical-records",
            "mirichoi0218/insurance",
            "protobioengineering/mit-bih-arrhythmia-database-modern-2023"
        ]

        # Display the table
        display_metadata(existing_datasets)

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

    elif data_source == "Upload file":

        # Allow both CSV and Excel file types
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                st.sidebar.success("CSV file uploaded!")
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                st.sidebar.success("Excel file uploaded!")
                st.session_state.df = pd.read_excel(uploaded_file)

            st.session_state.dataset_name = uploaded_file.name  # store actual filename to session state
            st.write(st.session_state.df)

def perform_knn_imputation():
    """Perform KNN imputation on select columns that meet criteria for having at least 1 missing value."""
    # Select columns with at least 1 missing value
    columns_with_nan = [col for col in st.session_state.df.columns if st.session_state.df[col].isnull().sum() > 0]
    if columns_with_nan:
        st.sidebar.markdown("### KNN Imputation")
        st.sidebar.markdown("The following columns have missing values and will be imputed using KNN:")
        st.sidebar.write(columns_with_nan)

        # Perform KNN imputation
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        st.session_state.df[columns_with_nan] = imputer.fit_transform(st.session_state.df[columns_with_nan])
        st.sidebar.success("KNN imputation completed!")
    else:
        st.sidebar.info("No columns with missing values found for KNN imputation.")

def add_column_based_on_conditions():
    st.subheader("Add Column Based on Conditions")
    condition_columns = st.multiselect("Select columns for conditions:", st.session_state.temp_df.columns)
    
    if not condition_columns:
        st.warning("No columns selected. Please select at least one column to avoid errors.")
        return
    
    conditions = []
    outputs = []
    
    for col in condition_columns:
        col_dtype = st.session_state.temp_df[col].dtype
    
        # A dictionary to store conditions and their corresponding output values for each column
        conditions_dict = {}
            
        # For numeric types
        if np.issubdtype(col_dtype, np.number):
            conditions_dict = handle_numeric_conditions(col, conditions_dict)
        else:
            st.warning(f"No predefined conditions for data type {col_dtype}")
    return conditions_dict

def handle_numeric_conditions(col, conditions_dict):
    # Create a condition-input mechanism for the selected column
    num_conditions = st.session_state.get("num_conditions", 1)
    
    for i in range(num_conditions):
        condition = st.text_input(f"Condition {i+1} for {col} (e.g., > 50):")
        output_value = st.text_input(f"Output value for condition {condition}:")
        
        # Validate the condition and output value before storing them
        try:
            dummy_df = pd.DataFrame({col: [0]})
            dummy_df.query(f"{col} {condition}")
            conditions_dict[condition] = float(output_value)
        except:
            st.warning(f"Invalid condition {condition} or output value {output_value} for column {col}.")
    
    # Allow user to add more conditions
    if st.button("+ Add Another Condition"):
        st.session_state.num_conditions += 1
        
    # Apply conditions to the dataframe
    else_output_value = st.text_input(f"Default output value if NO conditions are met:")
    
    try:
        # Allow user to name the new column, otherwise let the default occur
        new_col_name = st.text_input("Enter new column name:", f"{col}_new")
        
        # Check if the new column already exists in the DataFrame
        if new_col_name not in st.session_state.temp_df.columns:
            # Initializing the new column with the default value
            st.session_state.temp_df[new_col_name] = float(else_output_value)
        
        for condition, value in conditions_dict.items():
            condition_str = f"`{col}` {condition}"
            filtered_df = st.session_state.temp_df.query(condition_str)
            st.session_state.temp_df.loc[st.session_state.temp_df.index.isin(filtered_df.index), new_col_name] = value
        
        # Ask for user confirmation before renaming the column
        if st.button("Confirm column name"):
            st.write(f"Column {new_col_name} added to the DataFrame.")
        
    except Exception as e:
        st.warning(f"An error occurred while processing the conditions. Error: {e}")
    return conditions_dict

def save_changes():
    # Button to save changes
    if st.button("Save Changes"):
        st.session_state.df = st.session_state.temp_df.copy()
        st.success("Changes saved successfully!")
        st.write("Modified DataFrame:")
        st.write(st.session_state.df.head())
        st.info(f"Changes made to the DataFrame: \n\nAdded columns: {list(set(st.session_state.df.columns) - set(st.session_state.temp_df.columns))}")

def download_processed_data():
    # Download Processed Data
    st.subheader("Download Processed Data")
    if st.button("Download"):
        csv = st.session_state.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
def transform():
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("DataFrame not initialized. Please load the data first.")
        return

    # Create a temporary DataFrame to apply transformations
    if 'temp_df' not in st.session_state:
        st.session_state.temp_df = st.session_state.df.copy()

    perform_knn_imputation()
    conditions_dict = add_column_based_on_conditions()

    save_changes()
    download_processed_data()

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Go to Data section to start")
