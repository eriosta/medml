from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Define the number of samples


# Define the gold standard associations
gold_standard = [
    ('high', 0, 1, 0, 0, 'perihilar', 'unilateral', 'cad', 'acute', 'nstmi', 'pulmonary edema'),
    ('high', 1, 0, 1, 1, 'dependent', 'bilateral', 'chf', 'acute', 'sob', 'pulmonary edema'),
    ('equal', 1, 0, 1, 1, 'diffuse', 'bilateral', 'esrd', 'chronic', 'sob', 'pulmonary edema'),
    ('equal', 0, 1, 0, 0, 'diffuse', 'bilateral', 'cancer', 'chronic', 'sob', 'metastasis'),
    ('high', 1, 0, 1, 1, 'diffuse', 'unilateral', 'cancer', 'acute', 'aggresive thoracentesis', 'pulmonary edema')
]

# Convert gold_standard list to DataFrame
gold_standard_df = pd.DataFrame(gold_standard, columns=['ratio_upperlobe_lowerlobe_vessel', 
                                                        'is_increased_intestitial_markings', 
                                                        'is_distinct_pulmonary_vasculature',
                                                        'is_peribronchial_cuffing', 
                                                        'is_thickened_interlobular_fissures',
                                                        'regional_distribution_abnormality',
                                                        'laterality_abnormality', 'history', 'acuity', 'indication', 'rad_finding'])

# Duplicate each row a random number of times (between 1 and 20)
gold_standard_df = gold_standard_df.loc[gold_standard_df.index.repeat(np.random.randint(1, 21, size=len(gold_standard_df)))]

gold_standard_df['target'] = gold_standard_df['rad_finding'].apply(lambda x: 1 if x == 'pulmonary edema' else 0)
gold_standard_df = gold_standard_df.drop('rad_finding', axis=1)

# Define the target variable
y = gold_standard_df['target']

# Define the feature variables
X = gold_standard_df.drop('target', axis=1)

# Apply get_dummies to perform one-hot encoding
X = pd.get_dummies(X)

# Apply SMOTE to generate synthetic data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

