import streamlit as st
from pandas_profiling import ProfileReport

def st_profile_report(report):
    with st.spinner('Rendering report...'):
        st.components.v1.html(report.to_html(), height=600, width=800)

def generate_eda(df):
    if 'df' in st.session_state:
        df = st.session_state.df
        st.write("Generating EDA report...")
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        profile = ProfileReport(df, title="Automated EDA Report", explorative=True)
        progress_bar.progress(50)
    
        st_profile_report(profile)
        progress_bar.progress(75)
    
        profile.to_file("eda_report.html")
        with open("eda_report.html", "rb") as f:
            report_data = f.read()
        progress_bar.progress(100)
        return report_data
    else:
        st.warning("Please load source data under **Data** first")
