# https://qims.amegroups.org/article/view/9594/pdf

from faker import Faker
import numpy as np
import pandas as pd

fake = Faker()

# Define sample sizes
n_normal = 20
n_glaucoma = 21

# Normal Group
normal_age = np.random.normal(68.1, 7.3, n_normal)
normal_iop = np.random.normal(13.6, 3.5, n_normal)
normal_gender = np.random.choice(['Male', 'Female'], n_normal, p=[12/20, 8/20])
normal_sbp = np.random.normal(136.0, 21.1, n_normal)
normal_dbp = np.random.normal(82.8, 13.7, n_normal)
normal_mopp = np.random.normal(53.1, 9.2, n_normal)
normal_dm = [0] * n_normal
normal_htn = np.random.choice([1, 0], n_normal, p=[8/20, 12/20])
normal_systemic_med = np.random.choice([1, 0], n_normal, p=[9/20, 11/20])
normal_ocular_med = [0] * n_normal
normal_rnfl = np.random.normal(89.7, 9.9, n_normal)
normal_cdr = np.random.normal(0.44, 0.18, n_normal)
normal_rim = np.random.normal(1.30, 0.16, n_normal)

# Glaucoma Group
glaucoma_age = np.random.normal(62.9, 11.4, n_glaucoma)
glaucoma_iop = np.random.normal(15.0, 4.0, n_glaucoma)
glaucoma_gender = np.random.choice(['Male', 'Female'], n_glaucoma, p=[11/21, 10/21])
glaucoma_sbp = np.random.normal(127.1, 17.6, n_glaucoma)
glaucoma_dbp = np.random.normal(79.3, 11.6, n_glaucoma)
glaucoma_mopp = np.random.normal(48.1, 9.1, n_glaucoma)
glaucoma_dm = np.random.choice([1, 0], n_glaucoma, p=[4/21, 17/21])
glaucoma_htn = np.random.choice([1, 0], n_glaucoma, p=[4/21, 17/21])
glaucoma_systemic_med = np.random.choice([1, 0], n_glaucoma, p=[4/21, 17/21])
glaucoma_ocular_med = np.random.choice([1, 0], n_glaucoma, p=[18/21, 3/21])
glaucoma_rnfl = np.random.normal(66.5, 8.5, n_glaucoma)
glaucoma_cdr = np.random.normal(0.71, 0.11, n_glaucoma)
glaucoma_rim = np.random.normal(0.82, 0.19, n_glaucoma)

# Combine data
data = {
    'Group': ['Normal'] * n_normal + ['Glaucoma'] * n_glaucoma,
    'Age': np.concatenate([normal_age, glaucoma_age]),
    'IOP': np.concatenate([normal_iop, glaucoma_iop]),
    'Gender': np.concatenate([normal_gender, glaucoma_gender]),
    'SBP': np.concatenate([normal_sbp, glaucoma_sbp]),
    'DBP': np.concatenate([normal_dbp, glaucoma_dbp]),
    'MOPP': np.concatenate([normal_mopp, glaucoma_mopp]),
    'DM': np.concatenate([normal_dm, glaucoma_dm]),
    'HTN': np.concatenate([normal_htn, glaucoma_htn]),
    'SystemicMed': np.concatenate([normal_systemic_med, glaucoma_systemic_med]),
    'OcularMed': np.concatenate([normal_ocular_med, glaucoma_ocular_med]),
    'RNFL': np.concatenate([normal_rnfl, glaucoma_rnfl]),
    'CDR': np.concatenate([normal_cdr, glaucoma_cdr]),
    'RimArea': np.concatenate([normal_rim, glaucoma_rim])
}

df = pd.DataFrame(data)

(df).to_csv('glaucoma.csv')