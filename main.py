import streamlit as st
from data import setup_kaggle_env_variables, download_kaggle_dataset
from eda import generate_eda
import pandas as pd
import os

st.title("Kaggle Dataset Streamlit App")

if 'df' not in st.session_state:
    st.session_state.df = None

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
            report_data = generate_eda(st.session_state.df)
            st.sidebar.download_button(
                label="Download EDA Report",
                data=report_data,
                file_name="eda_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
    else:
        st.warning("Please upload a dataset first under 'Dataset Setup'.")
