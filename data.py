import os
import subprocess
import pandas as pd
import streamlit as st
import base64

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

    data_source = st.sidebar.radio("Choose Data Source", ["Upload file","Kaggle"])

    if data_source == "Kaggle":

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
            st.write(st.session_state.df.head())

def transform():
    if "df" in st.session_state:

        # Create a temporary DataFrame to apply transformations
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df = st.session_state.df.copy()
            
        # Filtering
        st.subheader("Filter Data")
        filter_column = st.selectbox("Select column to filter on:", st.session_state.df.columns)
        min_value = st.number_input(f"Minimum value for {filter_column}")
        max_value = st.number_input(f"Maximum value for {filter_column}")
        st.session_state.df = st.session_state.df[st.session_state.df[filter_column].between(min_value, max_value)]
        st.write("Filtered Data:")
        st.write(st.session_state.df.head())
    
        # Transformations
        st.subheader("Transform Data")
        cols_to_transform = st.multiselect("Select columns to transform:", st.session_state.df.columns)
        transform_type = st.selectbox("Transformation type:", ["Log", "Square root", "Custom (x^2)"])
        if st.button("Apply Transformation"):
            for col in cols_to_transform:
                if transform_type == "Log":
                    st.session_state.df[col] = np.log1p(st.session_state.df[col])
                elif transform_type == "Square root":
                    st.session_state.df[col] = np.sqrt(st.session_state.df[col])
                else:
                    st.session_state.df[col] = st.session_state.df[col]**2
            st.write("Transformed Data:")
            st.write(st.session_state.df.head())
    
        # Add Columns
        st.subheader("Add New Column")
        new_col_name = st.text_input("New column name:")
        new_col_formula = st.text_input("Formula (e.g., col1 + col2):")
        if st.button("Add Column"):
            df[new_col_name] = df.eval(new_col_formula)
            st.write("Data with New Column:")
            st.write(df.head())
    
        # Remove Columns
        st.subheader("Remove Columns")
        cols_to_remove = st.multiselect("Select columns to remove:", df.columns)
        if st.button("Remove Selected Columns"):
            df.drop(cols_to_remove, axis=1, inplace=True)
            st.write("Data after Column Removal:")
            st.write(df.head())
    
        # One-Hot Encoding
        st.subheader("One-Hot Encoding")
        cols_to_encode = st.multiselect("Select categorical columns to one-hot encode:", df.columns)
        if st.button("One-Hot Encode"):
            df = pd.get_dummies(df, columns=cols_to_encode)
            st.write("One-Hot Encoded Data:")
            st.write(df.head())
    
        # Button to save changes
        if st.button("Save Changes"):
            st.session_state.df = st.session_state.temp_df.copy()
            st.success("Changes saved successfully!")

        # Download Processed Data
        st.subheader("Download Processed Data")
        if st.button("Download"):
            csv = st.session_state.df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Go to Data section to start")
