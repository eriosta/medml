# https://qims.amegroups.org/article/view/117515/pdf

from faker import Faker
import numpy as np
import pandas as pd

fake = Faker()

n_without_IVH = 73
n_with_IVH = 41

# Without IVH Group
without_IVH_data = {
    'Age': np.random.normal(64.2, 13.7, n_without_IVH),
    'Gender': np.random.choice(['Male', 'Female'], n_without_IVH, p=[0.767, 0.233]),
    'Hypertension': np.random.choice([1, 0], n_without_IVH, p=[0.849, 0.151]),
    'Diabetes mellitus': np.random.choice([1, 0], n_without_IVH, p=[0.205, 0.795]),
    'Previous stroke': np.random.choice([1, 0], n_without_IVH, p=[0.247, 0.753]),
    'Antiplatelet use': np.random.choice([1, 0], n_without_IVH, p=[0.169, 0.831]),
    'Anticoagulant use': np.random.choice([1, 0], n_without_IVH, p=[0.028, 0.972]),
    'Admission GCS score': np.random.choice([15, 8], n_without_IVH, p=[0.942, 0.058]),
    'Admission SBP': np.random.uniform(153.0, 203.0, n_without_IVH),
    'Admission DBP': np.random.uniform(84.0, 111.0, n_without_IVH),
    'Admission glucose': np.random.uniform(5.9, 8.6, n_without_IVH),
    'Time to initial CT': np.random.uniform(1.0, 24.0, n_without_IVH),
    'Hematoma volume': np.random.uniform(1.5, 11.9, n_without_IVH),
    'Thalamic involvement': np.random.choice([1, 0], n_without_IVH, p=[0.342, 0.658]),
    'SI': np.random.normal(0.74, 0.08, n_without_IVH),
    'A/B ratio': np.random.uniform(1.3, 2.1, n_without_IVH),
    'LMA': np.random.uniform(10.0, 65.0, n_without_IVH)
}

# With IVH Group
with_IVH_data = {
    'Age': np.random.normal(66.6, 13.5, n_with_IVH),
    'Gender': np.random.choice(['Male', 'Female'], n_with_IVH, p=[0.659, 0.341]),
    'Hypertension': np.random.choice([1, 0], n_with_IVH, p=[0.854, 0.146]),
    'Diabetes mellitus': np.random.choice([1, 0], n_with_IVH, p=[0.293, 0.707]),
    'Previous stroke': np.random.choice([1, 0], n_with_IVH, p=[0.317, 0.683]),
    'Antiplatelet use': np.random.choice([1, 0], n_with_IVH, p=[0.146, 0.854]),
    'Anticoagulant use': np.random.choice([1, 0], n_with_IVH, p=[0.098, 0.902]),
    'Admission GCS score': np.random.choice([15, 8], n_with_IVH, p=[0.781, 0.219]),
    'Admission SBP': np.random.uniform(164.0, 189.5, n_with_IVH),
    'Admission DBP': np.random.uniform(80.0, 110.5, n_with_IVH),
    'Admission glucose': np.random.uniform(6.6, 9.5, n_with_IVH),
    'Time to initial CT': np.random.uniform(1.0, 5.5, n_with_IVH),
    'Hematoma volume': np.random.uniform(5.6, 14.6, n_with_IVH),
    'Thalamic involvement': np.random.choice([1, 0], n_with_IVH, p=[0.659, 0.341]),
    'SI': np.random.normal(0.69, 0.09, n_with_IVH),
    'A/B ratio': np.random.uniform(1.2, 1.8, n_with_IVH),
    'LMA': np.random.uniform(32.5, 78.0, n_with_IVH)
}

df_without_IVH = pd.DataFrame(without_IVH_data)
df_with_IVH = pd.DataFrame(with_IVH_data)

df = pd.concat([df_without_IVH, df_with_IVH], axis=0).reset_index(drop=True)

(df).to_csv('ivh.csv')