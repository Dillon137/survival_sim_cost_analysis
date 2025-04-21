import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

def generate_clinical_data(age_range=[55,85], sex=["Male", "Female"], diagnosis_stages=["III", "IV"], n_patients=200):
    # Patient Attributes
    ages = np.random.randint(age_range[0], age_range[1], size=n_patients)
    sexes = np.random.choice(sex, size=n_patients)
    performance_status = np.random.choice([0, 1, 2, 3], size=n_patients, p=[0.2, 0.5, 0.2, 0.1])
    smoking_status = np.random.choice(['Current', 'Former', 'Never'], size=n_patients, p=[0.4, 0.5, 0.1])
    stage_at_diagnosis = np.random.choice(["III", "IV"], size=n_patients, p=[0.3, 0.7])
    state = ['WI'] * n_patients

    smoking_dict = {
        'Never': 0.8,
        'Former': 1.0,
        'Current': 1.3
    }

    treatments = np.random.choice(['A', 'B'], size=n_patients)

    # Weibull parameters
    params = {
        ('III', 'A'): {'k': 1.5, 'lambda_ttp': 40, 'lambda_ttd': 60},
        ('III', 'B'): {'k': 1.5, 'lambda_ttp': 20, 'lambda_ttd': 40},
        ('IV', 'A'): {'k': 1.2, 'lambda_ttp': 24, 'lambda_ttd': 24},
        ('IV', 'B'): {'k': 1.2, 'lambda_ttp': 12, 'lambda_ttd': 12},
    }

    def adjust_lambda(base_lambda, age, perf_stat, smoking):
        age_adj = max(1, 1 + 0.01*(age-60))
        perf_adj = 1 + 0.1*(perf_stat-1)
        smoking_adj = smoking_dict[smoking]

        return base_lambda/(age_adj*perf_adj*smoking_adj)

    # Generate TTP and TTD
    time_to_progression = []
    time_to_death = []

    for patient in range(n_patients):
        k = params[(stage_at_diagnosis[patient], treatments[patient])]['k']
        base_lambda_ttp = params[(stage_at_diagnosis[patient], treatments[patient])]['lambda_ttp']
        lambda_ttp = adjust_lambda(base_lambda_ttp, ages[patient], performance_status[patient], smoking_status[patient])

        # If stage IV, progression is treated as death
        if stage_at_diagnosis[patient] == "IV":
            lambda_ttd = lambda_ttp
        else:
            base_lambda_ttd = params[(stage_at_diagnosis[patient], treatments[patient])]['lambda_ttd']
            lambda_ttd = adjust_lambda(base_lambda_ttd, ages[patient], performance_status[patient], smoking_status[patient])

        ttp = np.random.weibull(k)*lambda_ttp

        if stage_at_diagnosis[patient] == "IV":
            ttd = ttp
        else:
            ttd = np.random.weibull(k)*lambda_ttd

        ttp = min(ttp, ttd)  # Ensure TTP <= TTD
        time_to_progression.append(ttp)
        time_to_death.append(ttd)

    df = pd.DataFrame({
        'patient_id': np.arange(1, n_patients+1),
        'age': ages,
        'sex': sexes,
        'performance_status': performance_status,
        'smoking_status': smoking_status,
        'stage_at_diagnosis': stage_at_diagnosis,
        'state': state,
        'treatment': treatments,
        'time_to_progression': np.round(time_to_progression, 2),
        'time_to_death': np.round(time_to_death, 2)
    })

    return df


def run_simulation(patient_data, time_horizon=80, n_simulations=1000):
    # Initialize Markov states
    treatment_A = patient_data[patient_data["treatment"] == "A"]
    treatment_B = patient_data[patient_data["treatment"] == "B"]

    probs = []
    for treatment in [treatment_A, treatment_B]:
        treatment_probs = []

        # Stage III transition counts
        III_III_count = int((treatment[(treatment["stage_at_diagnosis"] == "III")]["time_to_progression"].apply(np.floor)).sum())

        III_IV_count = len(treatment[(treatment["stage_at_diagnosis"] == "III")
                            & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_progression"])

        III_death_count = len(treatment[(treatment["stage_at_diagnosis"] == "III")
                            & (treatment["time_to_progression"] == treatment["time_to_death"])]["time_to_progression"])

        III_total = III_III_count + III_IV_count + III_death_count


        # Stage IV transition counts
        stage_III_start = int((treatment[(treatment["stage_at_diagnosis"] == "III")
                                & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_death"]

                            - treatment[(treatment["stage_at_diagnosis"] == "III")
                                & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_progression"]).sum())

        IV_IV_count = int((treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"].apply(np.floor)).sum()) + stage_III_start  # Add with Stage III sum from progression time to death time

        IV_death_count = (len(treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"])
                          + len(treatment[(treatment["stage_at_diagnosis"] == "III")
                                & (treatment["time_to_progression"] != treatment["time_to_death"])]))

        IV_total = IV_IV_count + IV_death_count


        # Stage III transition probabilities
        III_III_prob = III_III_count / III_total
        III_IV_prob = III_IV_count / III_total
        III_death_prob = III_death_count / III_total
        treatment_probs.append([III_III_prob, III_IV_prob, III_death_prob])

        # Stage IV transition probabilities
        IV_IV_prob = IV_IV_count / IV_total
        IV_death_prob = IV_death_count / IV_total
        treatment_probs.append([0, IV_IV_prob, IV_death_prob])

        probs.append(treatment_probs)

    transition_probs = {
        "A": {
            "III": probs[0][0],
            "IV": probs[0][1]
        },

        "B": {
            "III": probs[1][0],
            "IV": probs[1][1]
        }
    }


    # Define health state costs per month (USD)
    costs = {
        "A": {
            "III": 1200,
            "IV": 2400,
            "Dead": 0
        },

        "B": {
            "III": 1000,
            "IV": 2000,
            "Dead": 0
        }
    }

    # Define QALYs per state per month
    qaly_weights = {
        "III": np.random.normal(0.85, 0.05),
        "IV": np.random.normal(0.5, 0.05),
        "Dead": 0
    }

    # Microsimulation Output
    def generate_paths(treatment):
        paths = []
        for _ in range(n_simulations):
            path = []
            state = np.random.choice(["III", "IV"], p=[0.3, 0.7])
            for _ in range(time_horizon):
                probs = transition_probs[treatment][state]
                next_state = np.random.choice(["III", "IV", "Dead"], p=probs)
                path.append(next_state)
                if next_state == "Dead":
                    path.extend(["Dead"] * (time_horizon - len(path)))
                    break
                state = next_state
            paths.append(path)
        return paths

    def compute_costs_qalys(paths, treatment):
        total_costs = []
        total_qalys = []
        for path in paths:
            cost = sum([costs[treatment][state] for state in path])
            qaly = sum([qaly_weights[state] for state in path]) / 12
            total_costs.append(cost)
            total_qalys.append(qaly)
        return np.array(total_costs), np.array(total_qalys)

    paths_new = generate_paths("A")
    costs_new, qalys_new = compute_costs_qalys(paths_new, "A")

    paths_ctrl = generate_paths("B")
    costs_ctrl, qalys_ctrl = compute_costs_qalys(paths_ctrl, "B")

    return paths_ctrl, costs_ctrl, qalys_ctrl, paths_new, costs_new, qalys_new