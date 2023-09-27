import streamlit as st
from data import data_run, transform
from eda import generate_eda
from models import train
from learn import show
from chat import llama2


class MEDMLApp:
    def __init__(self):
        self._init_session_state()
        st.sidebar.title("MEDML")
        self._display_dataset_info()
        self.navigate()

    def _init_session_state(self):
        if 'df' not in st.session_state:
            st.session_state.df = None
            st.session_state.dataset_name = None

    def _display_dataset_info(self):
        if st.session_state.dataset_name:
            st.sidebar.markdown(f"📊 **Loaded Dataset:** {st.session_state.dataset_name}")
        else:
            st.sidebar.info("No dataset currently loaded")

    def navigate(self):
        nav = st.sidebar.radio("Start Building", ["Get Started", "Data", "Models", "Extra"])

        if nav == "Get Started":
            GetStartedPage()
        elif nav == "Data":
            DataPage()
        elif nav == "Models":
            ModelsPage()
        elif nav == "Extra":
            ExtraPage()

class GetStartedPage:
    def __init__(self):
        self.render()

    def render(self):
        # Main Title and Logo (if available)
        # st.image("path_to_your_logo.png", use_column_width=True)  # Optional: Include your logo at the top
        st.title("MEDML: A Machine Learning Primer Built By Physicians For Physicians.")
        st.write("Embark on a hands-on journey into the world of machine learning, tailored for physicians.")

        # Section: Data Journey
        st.subheader("📊 Data Mastery")
        st.write("""
        - **Import & Understand**: Fetch datasets directly from Kaggle or upload your own in CSV format.
        - **Transform & Clean**: Prep your data for analysis with intuitive tools.
        - **Analyze**: Dive deep into your data with exploratory data analysis.
        """)

        # Section: Machine Learning
        st.subheader("🧠 Machine Learning Essentials")
        st.write("""
        - **Train & Test**: Build and evaluate classical ML models effortlessly.
        - **Optimization**: Tune hyperparameters for better accuracy.
        - **Explanation**: Understand model decisions with SHAP values.
        """)

        # Section: Learning & Advanced AI
        st.subheader("🚀 Elevate Your ML Knowledge")
        st.write("""
        - **Learn the Basics**: Comprehensive resources to solidify your foundation.
        - **Explore Generative AI**: Delve into advanced AI techniques with guided examples.
        """)

        # Disclaimer and Call to Action
        st.markdown("---")
        st.info("""
        Remember: MEDML is a tool to aid understanding. It doesn't replace clinical judgment or decision-making.
        Ready to dive in? Use the sidebar to start your ML journey!
        """)

        # Footer (Optional: Include contact info or other details)
        st.markdown("""
        Questions or feedback? [Contact us]().
        """)

class DataPage:
    def __init__(self):
        self.sub_navigation()

    def sub_navigation(self):
        data_page = st.sidebar.radio("Navigate", ['Source', 'Exploratory Analysis', 'Transformation'])
        if data_page == 'Source':
            data_run()
        elif data_page == "Exploratory Analysis":
            st.info("Under Development")
            # generate_eda()
        elif data_page == "Transformation":
            st.info("Under Development")

            # transform()

class ModelsPage:
    def __init__(self):
        self.sub_navigation()

    def sub_navigation(self):
        model_page = st.sidebar.radio("Navigate", ['Train & Evaluate', 'Explain'])
        if model_page == 'Train & Evaluate':
            if 'df' in st.session_state:
                train()
            else:
                st.warning("Please load data source under **Data** first")
        elif model_page == 'Explain':
            st.warning("Under construction.")

class ExtraPage:
    def __init__(self):
        self.sub_navigation()

    def sub_navigation(self):
        nav2 = st.sidebar.radio("Extra", ["Learn", "Generative AI"])
        if nav2 == "Learn":
            show()
        elif nav2 == "Generative AI":
            self.handle_generative_ai()

    def handle_generative_ai(self):
        navigation = st.sidebar.radio('Navigation', ['Llama2'])
        if navigation == 'Llama2':
            llama2()

# Run the main app
MEDMLApp()
