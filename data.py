import os
import subprocess
import pandas as pd
import streamlit as st

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
