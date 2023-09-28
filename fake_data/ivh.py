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
    'Gender': np.random.choice(['Male', 'Female'], n_without_IVH, p=[0.867, 0.133]),
    'Hypertension': np.random.choice([1, 0], n_without_IVH, p=[0.949, 0.051]),
    'Diabetes mellitus': np.random.choice([1, 0], n_without_IVH, p=[0.105, 0.895]),
    'Previous stroke': np.random.choice([1, 0], n_without_IVH, p=[0.147, 0.853]),
    'Antiplatelet use': np.random.choice([1, 0], n_without_IVH, p=[0.069, 0.931]),
    'Anticoagulant use': np.random.choice([1, 0], n_without_IVH, p=[0.018, 0.982]),
    'Admission GCS score': np.random.choice([15, 8], n_without_IVH, p=[0.982, 0.018]),
    'Admission SBP': np.random.uniform(153.0, 203.0, n_without_IVH),
    'Admission DBP': np.random.uniform(84.0, 111.0, n_without_IVH),
    'Admission glucose': np.random.uniform(5.9, 8.6, n_without_IVH),
    'Time to initial CT': np.random.uniform(1.0, 24.0, n_without_IVH),
    'Hematoma volume': np.random.uniform(1.5, 11.9, n_without_IVH),
    'Thalamic involvement': np.random.choice([1, 0], n_without_IVH, p=[0.242, 0.758]),
    'SI': np.random.normal(0.74, 0.08, n_without_IVH),
    'A/B ratio': np.random.uniform(1.3, 2.1, n_without_IVH),
    'LMA': np.random.uniform(10.0, 65.0, n_without_IVH),
    'is_ivh': [0]*n_without_IVH
}

# With IVH Group
with_IVH_data = {
    'Age': np.random.normal(66.6, 13.5, n_with_IVH),
    'Gender': np.random.choice(['Male', 'Female'], n_with_IVH, p=[0.559, 0.441]),
    'Hypertension': np.random.choice([1, 0], n_with_IVH, p=[0.954, 0.046]),
    'Diabetes mellitus': np.random.choice([1, 0], n_with_IVH, p=[0.393, 0.607]),
    'Previous stroke': np.random.choice([1, 0], n_with_IVH, p=[0.417, 0.583]),
    'Antiplatelet use': np.random.choice([1, 0], n_with_IVH, p=[0.246, 0.754]),
    'Anticoagulant use': np.random.choice([1, 0], n_with_IVH, p=[0.198, 0.802]),
    'Admission GCS score': np.random.choice([15, 8], n_with_IVH, p=[0.681, 0.319]),
    'Admission SBP': np.random.uniform(164.0, 189.5, n_with_IVH),
    'Admission DBP': np.random.uniform(80.0, 110.5, n_with_IVH),
    'Admission glucose': np.random.uniform(6.6, 9.5, n_with_IVH),
    'Time to initial CT': np.random.uniform(1.0, 5.5, n_with_IVH),
    'Hematoma volume': np.random.uniform(5.6, 14.6, n_with_IVH),
    'Thalamic involvement': np.random.choice([1, 0], n_with_IVH, p=[0.759, 0.241]),
    'SI': np.random.normal(0.69, 0.09, n_with_IVH),
    'A/B ratio': np.random.uniform(1.2, 1.8, n_with_IVH),
    'LMA': np.random.uniform(32.5, 78.0, n_with_IVH),
    'is_ivh': [1]*n_with_IVH
}

df_without_IVH = pd.DataFrame(without_IVH_data)
df_with_IVH = pd.DataFrame(with_IVH_data)

df = pd.concat([df_without_IVH, df_with_IVH], axis=0).reset_index(drop=True)

(df).to_csv('ivh.csv')