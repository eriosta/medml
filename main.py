import streamlit as st
import os
import pandas as pd
import subprocess
from pandas_profiling import ProfileReport

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

# To display the pandas profile report within streamlit
def st_profile_report(report):
    with st.spinner('Rendering report...'):
        st.components.v1.html(report.to_html(), height=600, width=800)

st.title("Kaggle Dataset Streamlit App")

# Check if 'df' is already in the session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Navigation
nav = st.sidebar.radio("Navigation", ["Dataset Setup", "EDA Report"])

if nav == "Dataset Setup":
    data_source = st.sidebar.radio("Choose your data source", ["Kaggle Dataset", "Upload your CSV"])

    if data_source == "Kaggle Dataset":
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
        dataset_path = dataset_choice if dataset_choice != "Custom Dataset" else st.sidebar.text_input("Enter Kaggle Dataset Path")

        if dataset_path:
            try:
                zip_name = download_kaggle_dataset(dataset_path)
                zip_path = os.path.join(zip_name)
                st.sidebar.success(f"{dataset_path} downloaded!")

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
            st.write(st.session_state.df.head())

elif nav == "EDA Report":
    if st.session_state.df is not None:
        try:
            st.write("Generating EDA report...")
            
            # Create a progress bar
            progress_bar = st.progress(0)

            # Update progress bar: start (You can update this with any logic)
            progress_bar.progress(25)

            profile = ProfileReport(st.session_state.df, title="Automated EDA Report", explorative=True)

            # Update progress bar: mid-way
            progress_bar.progress(50)

            st_profile_report(profile)

            # Update progress bar: almost done
            progress_bar.progress(75)

            # Allow user to download the report
            profile.to_file("eda_report.html")
            with open("eda_report.html", "rb") as f:
                report_data = f.read()
            st.sidebar.download_button(
                label="Download EDA Report",
                data=report_data,
                file_name="eda_report.html",
                mime="text/html"
            )

            # Update progress bar: complete
            progress_bar.progress(100)

        except Exception as e:
            st.error(f"Error generating report: {e}")
    else:
        st.warning("Please upload a dataset first under 'Dataset Setup'.")
