# Import necessary libraries
import streamlit as st
import pandas as pd

def show():

    selection = st.sidebar.radio("Go to", ["Which Model Do I Use?","Models 101"])

    if selection == "Which Model Do I Use?":
        # Title
        st.header('Which Model Do I Use?')
    
        # Input parameters
    
        st.markdown("""
        1. **Target Variable**: 
        - **Categorical (Classification)**: The output or label in your data is categorical (e.g., "Yes" or "No").
        - **Continuous (Regression)**: The output or label is continuous (e.g., predicting house prices).""")
    
        target_variable = st.selectbox('Target Variable:', ['Categorical (Classification)', 'Continuous (Regression)'])
    
        st.markdown("""
        2. **Missing Data Handling**: 
        - **Data Imputation**: Replacing missing data with substituted values.
        - **Use a Model that Handles Missing Data**: Algorithms like XGBoost or LightGBM handle missing values directly.
        """)
    
        missing_data_handling = st.selectbox('Missing Data Handling:', ['Data Imputation', 'Use a Model that Handles Missing Data'])
    
        st.markdown("""
        3. **Data Size**: 
        - **Small**: Fewer than 10,000 samples.
        - **Medium**: Between 10,0000 to 500,000 samples.
        - **Large**: More than 500,000 samples.
    
        It's important to note that these are rough numbers and can greatly depend on the feature dimensionality, the problem at hand, and the computational resources available. For instance, in some complex domains, even 10,000 samples might be considered large due to the richness of the data.
        """)
                    
        data_size = st.selectbox('Data Size:', ['Small', 'Medium', 'Large'])
    
        st.markdown("""
        4. **Number of Features**: 
        - **A Few**: 1 - 20 features (columns).
        - **Many**: More than 20 features.
    
        Again, these numbers can be context-dependent. For instance, in the context of a high-dimensional genomics dataset, even 100 features might be considered "a few". But for many typical datasets, having more than 20 features starts getting into the realm of requiring special attention to feature selection, dimensionality reduction, and other complexities.
        """)
    
        num_features = st.selectbox('Number of Features:', ['A Few', 'Many'])
    
        st.markdown("""
        5. **Collinearity Among Features**: 
    
        Collinearity refers to the situation in which two or more features (or predictor variables) in a dataset are highly correlated, meaning one can be linearly predicted from the others with significant accuracy. This can complicate the interpretation of a model and make it hard to determine the individual effect of predictors, as their effects are entangled with each other.
        """)
                    
        collinearity = st.selectbox('Collinearity Among Features:', ['Low Collinearity', 'High Collinearity'])
    
        st.markdown("""
        6. **Data Distribution**: 
        - **Linear Relationship**: The relationship between features and target is linear.
        - **Non-Linear Relationship**: The relationship is complex and doesn't follow a straight line.
        """)
                    
        data_distribution = st.selectbox('Data Distribution:', ['Linear Relationship', 'Non-Linear Relationship'])
    
        # Recommendations based on input
        st.subheader('Recommendations:')
    
        if target_variable == 'Categorical (Classification)':
            if data_size == 'Small' or data_size == 'Medium':
                st.write('Small or Medium Size Data: Logistic Regression, Support Vector Machines')
            else:
                st.write('Large Size Data: Random Forest, Gradient Boosting (like XGBoost), Neural Networks')
    
        elif target_variable == 'Continuous (Regression)':
            if data_size == 'Small' or data_size == 'Medium':
                st.write('Small or Medium Size Data: Linear Regression, Support Vector Regression')
            else:
                st.write('Large Size Data: Random Forest Regressor, Gradient Boosting Regressor (like XGBoost), Neural Network Regression')
    
        if missing_data_handling == 'Data Imputation':
            st.write('For categorical data, consider imputation using mode or most frequent category.')
            st.write('For continuous data, consider imputation using mean, median, or model-based methods like KNN imputation.')
        else:
            st.write('Consider using models like XGBoost or LightGBM which handle missing data directly.')
    
        if data_size == 'Small':
            st.write('Emphasize on simpler models like Logistic/Linear Regression or SVM and consider regularization (e.g., Lasso or Ridge Regression).')
        elif data_size == 'Medium':
            st.write('Consider ensemble methods like Bagging or Boosting. Decision Trees and Random Forests are also suitable.')
    
        if num_features == 'Many':
            st.write('Consider dimensionality reduction (like PCA) or models that perform implicit feature selection like Random Forest or XGBoost.')
    
        if collinearity == 'High Collinearity':
            st.write('Avoid linear models without regularization as they are sensitive to multicollinearity. Use models like Decision Trees or Random Forests. Also, consider regularization techniques like Ridge or Lasso regression.')
    
        if data_distribution == 'Linear Relationship':
            st.write('Linear regression (for regression tasks) or logistic regression (for classification tasks) may work well.')
        else:
            st.write('Consider methods like Decision Trees, Random Forests, Neural Networks, and SVM with non-linear kernels.')
    
        st.write('Always remember to cross-validate on your dataset and check multiple models for best performance.')

    if selection == "Models 101":
        st.header("Models")
    
        # Creating the dataframe excluding k-Nearest Neighbors (KNN)...
        data = {
            'Model': ['Logistic Regression', 'Support Vector Machines', 'Naive Bayes', 'Random Forest', 'Gradient Boosting (e.g. XGBoost)', 'Linear Regression', 'Neural Networks'],
            'Description': ['Statistical method for modeling the relationship between a categorical dependent variable and one or more independent variables.',
                            'Uses a hyperplane to classify data into two classes.',
                            'Based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features.',
                            'Ensemble of decision trees, each one trained on a random subset of the data.',
                            'Boosting technique which converts weak learners to strong learners.',
                            'Models the relationship between a continuous target variable and one or more independent variables.',
                            'Composed of layers of nodes or neurons where data is processed, transformed, and used for prediction.'],
            'Pros': ['Easy to implement; Provides probabilities for outcomes.',
                    'Effective in high dimensional spaces; Uses a subset of training points.',
                    'Fast and simple; Works well with small datasets.',
                    'Handles large datasets; Can model non-linear relationships.',
                    'Handles missing data; Often provides higher accuracy.',
                    'Simple and interpretable; Works well when relationship is linear.',
                    'Can model complex, non-linear relationships; Versatile with different architectures.'],
            'Cons': ['Not suitable for non-linear relationships; Assumes no multicollinearity.',
                    'Can be slow; Not directly applicable to multi-class classification.',
                    'Makes a strong assumption about feature independence; Can be outperformed by more advanced techniques.',
                    'Can be slow to train; Might overfit on noisy datasets.',
                    'Computationally intensive; Might overfit if not tuned properly.',
                    'Sensitive to outliers; Assumes linear relationship.',
                    'Requires a lot of data; Can be hard to interpret and debug.']
        }
    
        df = pd.DataFrame(data)
    
        # Display each model in a collapsible section
        for index, row in df.iterrows():
            with st.expander(row["Model"]):
                st.write("**Description:**", row["Description"])
                st.write("**Pros:**", row["Pros"])
                st.write("**Cons:**", row["Cons"])
