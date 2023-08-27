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

# You can then call the function as:
# existing_datasets_sample = ["user1/dataset1", "user2/dataset2"]  # Example datasets
# display_metadata(existing_datasets_sample)
