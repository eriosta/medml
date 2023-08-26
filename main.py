import streamlit as st
import os
import pandas as pd
from pandas_profiling import ProfileReport
import subprocess

BASE_DIR = 'workspaces/medml/'

def setup_kaggle_env_variables(username, key):
    """Set up environment variables for Kaggle."""
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

def download_kaggle_dataset(dataset_path):
    """Download and unzip the specified kaggle dataset using Kaggle CLI."""
    
    # Use subprocess to execute CLI commands
    download_command = f"kaggle datasets download {dataset_path}"
    download_result = subprocess.run(download_command, shell=True, capture_output=True, text=True)
    
    if download_result.returncode != 0:
        raise ValueError(download_result.stderr)

    # Extract dataset name from the dataset path
    dataset_name = dataset_path.split("/")[-1]
    unzip_command = f"unzip {dataset_name}.zip"
    unzip_result = subprocess.run(unzip_command, shell=True, capture_output=True, text=True)
    
    if unzip_result.returncode != 0:
        raise ValueError(unzip_result.stderr)

    # Return the CSV file name
    for file in unzip_result.stdout.split("\n"):
        if file.endswith(".csv"):
            return file.strip()

st.title("Kaggle Dataset Streamlit App")

# Sidebar content
st.sidebar.header("Kaggle Setup")
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
if dataset_choice == "Custom Dataset":
    dataset_path = st.sidebar.text_input("Enter Kaggle Dataset Path")
else:
    dataset_path = dataset_choice

csv_filename = None  # Initialize CSV file name to None

if dataset_path:
    try:
        csv_filename = download_kaggle_dataset(dataset_path)
        st.sidebar.success(f"{dataset_path} downloaded and extracted!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Main content
if csv_filename and st.sidebar.button("Generate EDA Report"):
    # Join the base directory with the csv filename to get the full path
    full_csv_path = os.path.join(BASE_DIR, csv_filename)
    
    df = pd.read_csv(full_csv_path)

    # Display the head of the DataFrame to ensure it's read correctly
    st.write(df.head())

    # Generate the EDA report
    profile = ProfileReport(df, title="Automated EDA Report", explorative=True)
    report_path = os.path.join(BASE_DIR, "eda_report.html")  # Full path for EDA report as well
    profile.to_file(report_path)

    # Use st.components to display the EDA report
    with open(report_path, 'r') as f:
        html_string = f.read()
    st.components.v1.html(html_string, width=800, height=600, scrolling=True)