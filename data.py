import os
import subprocess
import pandas as pd
import streamlit as st
import base64
import numpy as np
from sklearn.impute import KNNImputer

def data_run():

    data_source = st.sidebar.radio("Choose Data Source", ["Synthetic Data","Upload file"])
            
    if data_source == "Synthetic Data":

        fake_data_files = ["breast_cancer.csv", "glaucoma.csv", "iah.csv", "ivh.csv"]
        fake_data_choice = st.sidebar.selectbox("Choose a Fake Data File", fake_data_files)

        if fake_data_choice:
            try:
                st.session_state.df = pd.read_csv(f"fake_data/{fake_data_choice}")
                st.session_state.df.reset_index(drop=True, inplace=True)  # Drop the index
                if 'Unnamed: 0' in st.session_state.df.columns:
                    st.session_state.df.drop(columns=['Unnamed: 0'], inplace=True)  # Drop the column "Unnamed: 0"
                st.session_state.dataset_name = fake_data_choice.split(".")[0]  # store dataset name to session state
                st.write(st.session_state.df)

                # Match the selected dataset with its data dictionary from metadata.py
                from metadata import breast_cancer_data_dict
                if st.session_state.dataset_name == 'breast_cancer':
                    data_dict = breast_cancer_data_dict
                # Add other data dictionaries as needed
                # if st.session_state.dataset_name == 'other_dataset':
                #     data_dict = other_data_dict

                # Show the dictionary data as an expandable row for better UI
                for key, value in data_dict.items():
                    with st.expander(key):
                        st.text(value)

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

            st.session_state.df.reset_index(drop=True, inplace=True)  # Drop the index
            if 'Unnamed: 0' in st.session_state.df.columns:
                st.session_state.df.drop(columns=['Unnamed: 0'], inplace=True)  # Drop the column "Unnamed: 0"
            st.session_state.dataset_name = uploaded_file.name  # store actual filename to session state
            st.write(st.session_state.df)

def perform_knn_imputation():
    """Perform KNN imputation on select columns that meet criteria for having at least 1 missing value."""
    # Select columns with at least 1 missing value
    columns_with_nan = [col for col in st.session_state.df.columns if st.session_state.df[col].isnull().sum() > 0]
    if columns_with_nan:
        st.markdown("### KNN Imputation")
        st.markdown("The following columns have missing values and can be imputed using KNN:")
        st.write(columns_with_nan)

        # Let the user select which columns to impute
        columns_to_impute = st.multiselect("Select columns to impute", columns_with_nan)

        if columns_to_impute:
            for col in columns_to_impute:
                try:
                    # Perform KNN imputation
                    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
                    st.session_state.temp_df[col] = imputer.fit_transform(st.session_state.df[[col]])
                    st.success(f"KNN imputation completed for {col}!")
                except ValueError:
                    st.warning(f"Cannot perform KNN imputation on {col} as it is not numeric. Please select another column.")
        else:
            st.info("No columns selected for KNN imputation.")
    else:
        st.info("No columns with missing values found for KNN imputation.")

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

