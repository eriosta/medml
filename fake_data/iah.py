# Cite this article as: Chen B, Yang S, Lyu G, Cheng X, Chen M,
# Xu J. A nomogram for predicting the risk of intra-abdominal
# hypertension in critically ill patients based on ultrasound and
# clinical data. Quant Imaging Med Surg 2023;13(10):7041-7051.
# doi: 10.21037/qims-23-

from faker import Faker
import numpy as np
import pandas as pd

fake = Faker()

n_non_IAH = 44
n_IAH = 45

# Non-IAH Group
non_IAH_data = {
    'Age': np.random.randint(56, 79, n_non_IAH),
    'Gender': np.random.choice(['Male', 'Female'], n_non_IAH, p=[0.8182, 0.1818]),
    'SOFA score': np.random.randint(2, 3, n_non_IAH),
    'PaO2 /FiO2': np.random.uniform(242.75, 335.25, n_non_IAH),
    'Platelet': np.random.uniform(140.75, 262.50, n_non_IAH),
    'Bilirubin': np.random.uniform(11.90, 24.53, n_non_IAH),
    'MAP': np.random.normal(80.5, 12.38, n_non_IAH),
    'GCS': [15] * n_non_IAH,  # Constant
    'Creatinine': np.random.uniform(57.45, 126.88, n_non_IAH),
    'RRRI': np.random.uniform(0.58, 0.67, n_non_IAH),
    'RKE': np.random.uniform(5.90, 9.36, n_non_IAH),
    'RDTR': np.random.uniform(0.17, 0.23, n_non_IAH),
    'RDE': np.random.uniform(18.25, 27.34, n_non_IAH),
    'Lac': np.random.uniform(0.73, 1.50, n_non_IAH),
    'Heart rate': np.random.normal(95.18, 17.85, n_non_IAH),
    'Breathing': np.random.uniform(16.00, 20.75, n_non_IAH)
}

# IAH Group
IAH_data = {
    'Age': np.random.randint(44, 73, n_IAH),
    'Gender': np.random.choice(['Male', 'Female'], n_IAH, p=[0.8222, 0.1778]),
    'SOFA score': np.random.randint(2, 4, n_IAH),
    'PaO2 /FiO2': np.random.uniform(244.00, 314.00, n_IAH),
    'Platelet': np.random.uniform(145.50, 272.50, n_IAH),
    'Bilirubin': np.random.uniform(13.25, 45.10, n_IAH),
    'MAP': np.random.normal(82.89, 12.49, n_IAH),
    'GCS': [15] * n_IAH,  # Constant
    'Creatinine': np.random.uniform(60.35, 104.90, n_IAH),
    'RRRI': np.random.uniform(0.70, 0.73, n_IAH),
    'RKE': np.random.uniform(5.65, 9.12, n_IAH),
    'RDTR': np.random.uniform(0.10, 0.14, n_IAH),
    'RDE': np.random.uniform(20.43, 29.60, n_IAH),
    'Lac': np.random.uniform(0.95, 1.50, n_IAH),
    'Heart rate': np.random.normal(91.76, 18.39, n_IAH),
    'Breathing': np.random.uniform(16.00, 22.50, n_IAH)
}

df_non_IAH = pd.DataFrame(non_IAH_data)
df_IAH = pd.DataFrame(IAH_data)

df = pd.concat([df_non_IAH, df_IAH], axis=0).reset_index(drop=True)

(df).to_csv('iah.csv')