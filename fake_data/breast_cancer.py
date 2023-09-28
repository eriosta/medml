# https://doi.org/10.1148/radiol.223077

import pandas as pd
import numpy as np

# Define the number of samples for each category
n_samples_main = 10

# Generate main synthetic data
main_data = {
    'Age': np.random.choice(range(48, 61), n_samples_main),
    'Race and/or ethnicity': np.random.choice(['American Indian', 'Asian', 'Black', 'Hispanic', 'Other', 'White and non-Hispanic'], n_samples_main, p=[0.001, 0.034, 0.018, 0.008, 0.021, 0.918]),
    'Mammographic breast density': np.random.choice(['Fatty', 'Scattered fibroglandular', 'Heterogeneously dense', 'Extremely dense'], n_samples_main, p=[0.014, 0.315, 0.556, 0.115]),
    'Personal history of breast cancer': np.random.choice(['No', 'Yes'], n_samples_main, p=[0.702, 0.298]),
    'Family history of breast cancer': np.random.choice(['No', 'Yes'], n_samples_main, p=[0.231, 0.769]),
    'Screening indication': np.random.choice(['Genetic mutation', 'History of chest radiation', 'Personal history of breast cancer', 'Family history of breast cancer', 'Personal history of high-risk lesion', 'Other'], n_samples_main, p=[0.072, 0.015, 0.254, 0.408, 0.234, 0.017]),
    'DL 5-y risk': np.random.choice(['Increased risk (score ≥ 2.3)', 'Not increased risk (score < 2.3)'], n_samples_main, p=[0.538, 0.462])
}

# Generating risk data with the same length as main data
main_data['TC 5-y risk'] = np.random.choice(['Increased risk (score ≥ 1.67%)', 'Not increased risk (score < 1.67%)'], n_samples_main, p=[0.790, 0.210])
main_data['NCI BCRAT 5-y risk'] = np.random.choice(['Increased risk (score ≥ 1.67%)', 'Not increased risk (score < 1.67%)'], n_samples_main, p=[0.771, 0.229])
main_data['TC lifetime risk'] = np.random.choice(['High risk (score ≥ 20%)', 'Not high risk (score < 20%)'], n_samples_main, p=[0.518, 0.482])
main_data['NCI BCRAT lifetime risk'] = np.random.choice(['High risk (score ≥ 20%)', 'Not high risk (score < 20%)'], n_samples_main, p=[0.363, 0.637])

# Create main dataframe
df_main = pd.DataFrame(main_data)

# Continuation from previous code

# Generate unique patient_ids
unique_ids = set()

while len(unique_ids) < n_samples_main:
    unique_ids.add(np.random.randint(10000000, 99999999))

df_main['patient_id'] = list(unique_ids)

(df_main).to_csv("fake_data/breast_cancer.csv")

