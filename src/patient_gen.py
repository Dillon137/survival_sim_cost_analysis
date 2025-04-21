import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of synthetic patients
n_patients = 200

# 1. Patient Demographics
ages = np.random.randint(55, 66, size=n_patients)
sexes = np.random.choice(['Male', 'Female'], size=n_patients)
performance_status = np.random.choice([0, 1, 2, 3], size=n_patients, p=[0.2, 0.5, 0.2, 0.1])
smoking_status = np.random.choice(['Current', 'Former', 'Never'], size=n_patients, p=[0.4, 0.5, 0.1])
stage_at_diagnosis = np.random.choice(["III", "IV"], size=n_patients, p=[0.3, 0.7])
state = ['WI'] * n_patients

# Treatment Assignment
treatments = np.random.choice(['A', 'B'], size=n_patients)

"""
###### Exponential Estimate. Might want to generate multiple examples of progression #########
time_to_progression = np.where(
    treatments == 'A',
    np.random.exponential(scale=ttp_a, size=n_patients),
    np.random.exponential(scale=ttp_b, size=n_patients)
)

time_to_death = np.where(
    treatments == 'A',
    np.random.exponential(scale=ttd_a, size=n_patients),
    np.random.exponential(scale=ttd_b, size=n_patients)
)
"""


# Weibull parameters by (stage, treatment)
params = {
    ('III', 'A'): {'k': 1.5, 'lambda_ttp': 40, 'lambda_ttd': 60},
    ('III', 'B'): {'k': 1.5, 'lambda_ttp': 20, 'lambda_ttd': 40},
    ('IV', 'A'): {'k': 1.5, 'lambda_ttp': 24,  'lambda_ttd': 24},
    ('IV', 'B'): {'k': 1.5, 'lambda_ttp': 12, 'lambda_ttd': 12},
}


def adjust_lambda(base_lambda, age, performance_stat, smoking):
    age_adj = max(1 + 0.01 * (age - 60), 1)  # small increase per year over 60
    perf_adj = 1 + 0.1 * performance_stat  # worse PS, worse prognosis
    smoking_adj = {
        'Never': 0.9,
        'Former': 1.0,
        'Current': 1.1
    }[smoking]

    return base_lambda / (age_adj * perf_adj * smoking_adj)


# Generate TTP and TTD
time_to_progression = []
time_to_death = []

for stage, treatment in zip(stage_at_diagnosis, treatments):
    k = params[(stage, treatment)]['k']
    lambda_ttp = params[(stage, treatment)]['lambda_ttp']

    # If stage IV, progression is treated as death
    if stage == "IV": lambda_ttd = lambda_ttp
    else: lambda_ttd = params[(stage, treatment)]['lambda_ttd']

    ttp = np.random.weibull(k) * lambda_ttp

    # If stage IV, progression is treated as death
    if stage == "IV": ttd = ttp
    else: ttd = np.random.weibull(k) * lambda_ttd

    ttp = min(ttp, ttd)  #Ensure TTP <= TTD
    time_to_progression.append(ttp)
    time_to_death.append(ttd)


# Event Indicators
# progressed = time_to_progression < time_to_death
# died = np.ones(n_patients, dtype=bool)  # Assume all eventually die


df = pd.DataFrame({
    'patient_id': np.arange(1, n_patients + 1),
    'age': ages,
    'sex': sexes,
    'performance_status': performance_status,
    'smoking_status': smoking_status,
    'stage_at_diagnosis': stage_at_diagnosis,
    'state': state,
    'treatment': treatments,
    'time_to_progression': np.round(time_to_progression, 2),
    'time_to_death': np.round(time_to_death, 2)
    # 'progressed': progressed,
    # 'died': died
})

# Save to CSV
df.to_csv("../data/synthetic_lung_cancer_trial_data.csv", index=False)

print("Synthetic dataset saved as 'synthetic_lung_cancer_trial_data.csv'")