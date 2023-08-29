import os
import subprocess
import pandas as pd
import streamlit as st
import base64
import numpy as np

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

def transform():
    # st.warning("Under construction")
    if "df" in st.session_state:
        # Create a temporary DataFrame to apply transformations
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df = st.session_state.df.copy()

        # Add Column Based on Conditions
        st.subheader("Add Column Based on Conditions")
        condition_columns = st.multiselect("Select columns for conditions:", st.session_state.temp_df.columns)
    
        if condition_columns:
            conditions = []
            outputs = []
    
            for col in condition_columns:
                col_dtype = st.session_state.temp_df[col].dtype
    
                # A dictionary to store conditions and their corresponding output values for each column
                conditions_dict = {}
                
                # For numeric types
                if np.issubdtype(col_dtype, np.number):
                    
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
                        
                        # Initializing the new column with the default value
                        st.session_state.temp_df[new_col_name] = float(else_output_value)
                        
                        for condition, value in conditions_dict.items():
                            condition_str = f"`{col}` {condition}"
                            filtered_df = st.session_state.temp_df.query(condition_str)
                            st.session_state.temp_df.loc[st.session_state.temp_df.index.isin(filtered_df.index), new_col_name] = value
                        
                        st.write(f"Column {new_col_name} added to the DataFrame.")
                        st.write(st.session_state.temp_df.head())
                        
                    except Exception as e:
                        st.warning(f"An error occurred while processing the conditions. Error: {e}")
                   
                else:
                    st.warning(f"No predefined conditions for data type {col_dtype}")
        
        # Transformations
        st.subheader("Transform Data")
        cols_to_transform = st.multiselect("Select columns to transform:", st.session_state.df.columns)
        
        if cols_to_transform:
            # Detect the data type of the first selected column (for simplicity)
            col_dtype = st.session_state.df[cols_to_transform[0]].dtype
    
            if np.issubdtype(col_dtype, np.number):  # if numeric
                transform_type = st.selectbox("Transformation type:", ["Log", "Square root", "Custom (x^2)"])
                if st.button("Apply Numeric Transformation"):
                    for col in cols_to_transform:
                        # Allow user to name the new column, otherwise let the default occur
                        new_col_name = st.text_input("Enter new column name:", f"{col}_transformed")
                        if transform_type == "Log":
                            st.session_state.temp_df[new_col_name] = np.log1p(st.session_state.temp_df[col])
                        elif transform_type == "Square root":
                            st.session_state.temp_df[new_col_name] = np.sqrt(st.session_state.temp_df[col])
                        else:
                            st.session_state.temp_df[new_col_name] = st.session_state.temp_df[col]**2
    
            st.write("Transformed Data:")
            st.write(st.session_state.temp_df.head())
        
        # One-Hot Encoding
        st.subheader("One-Hot Encoding")
        cols_to_encode = st.multiselect("Select categorical columns to one-hot encode:", st.session_state.temp_df.columns)
        if st.button("One-Hot Encode"):
            st.session_state.temp_df = pd.get_dummies(st.session_state.temp_df, columns=cols_to_encode)
            st.write("One-Hot Encoded Data:")
            st.write(st.session_state.temp_df.head())
        
        # Button to save changes
        if st.button("Save Changes"):
            st.session_state.df = st.session_state.temp_df.copy()
            st.success("Changes saved successfully!")
            st.write("Modified DataFrame:")
            st.write(st.session_state.df.head())
            st.info(f"Changes made to the DataFrame: \n\nAdded columns: {list(set(st.session_state.df.columns) - set(st.session_state.temp_df.columns))}\n\nTransformed columns: {cols_to_transform}\n\nOne-Hot Encoded columns: {cols_to_encode}")
        
        # Download Processed Data
        st.subheader("Download Processed Data")
        if st.button("Download"):
            csv = st.session_state.df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Go to Data section to start")
