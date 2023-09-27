import streamlit as st
import json
from ydata_profiling import ProfileReport
from metadata import breast_cancer_data_dict

def st_profile_report(report):
    with st.spinner('Rendering report...'):
        st.components.v1.html(report.to_html(), height=600, width=800)

def generate_eda():
    if 'df' in st.session_state:
        df = st.session_state.df
        st.write("Generating EDA report...")
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        profile = ProfileReport(df)
        progress_bar.progress(50)
    
        st_profile_report(profile)
        progress_bar.progress(75)
    
        profile.to_file("eda_report.html")
        with open("eda_report.html", "rb") as f:
            report_data = f.read()
        progress_bar.progress(100)
        
        # Add dataset description and metadata
        report = df.profile_report(
            title="Breast Cancer Data",
            dataset={
                "description": "This profiling report was generated using the breast cancer dataset.",
                "copyright_holder": "Your Company Name",
                "copyright_year": 2022,
                "url": "http://www.yourcompany.com/datasets/breast_cancer.csv",
            },
            variables={
                "descriptions": breast_cancer_data_dict
            }
        )
        report.to_file("eda_report.html")
        return report_data
    else:
        st.warning("Please load source data under **Data** first")


