import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lifelines import KaplanMeierFitter

np.random.seed(42)

# Number of trial patients
n_patients = 200

# Patient Attributes
ages = np.random.randint(55, 66, size=n_patients)
sexes = np.random.choice(['Male', 'Female'], size=n_patients)
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
    age_adj = max(1, 1 + 0.01 * (age - 60))
    perf_adj = 1 + 0.1 * (perf_stat - 1)

    smoking_adj = smoking_dict[smoking]

    return base_lambda / (age_adj * perf_adj * smoking_adj)


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

    ttp = np.random.weibull(k) * lambda_ttp

    if stage_at_diagnosis[patient] == "IV":
        ttd = ttp
    else:
        ttd = np.random.weibull(k) * lambda_ttd

    ttp = min(ttp, ttd)  # Ensure TTP <= TTD
    time_to_progression.append(ttp)
    time_to_death.append(ttd)

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
})

patient_data = df

# Initialize Markov states
states = ['III', 'IV', 'Death']
transition_counts = pd.DataFrame(0, index=states, columns=states)

treatment_A = patient_data[patient_data["treatment"] == "A"]
treatment_B = patient_data[patient_data["treatment"] == "B"]

probs = []
for treatment in [treatment_A, treatment_B]:
    treatment_probs = []

    # Stage III transition counts
    III_III_count = int(
        (treatment[(treatment["stage_at_diagnosis"] == "III")]["time_to_progression"].apply(np.floor)).sum())
    III_IV_count = len(treatment[(treatment["stage_at_diagnosis"] == "III") & (
                treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_progression"])
    III_death_count = len(treatment[(treatment["stage_at_diagnosis"] == "III") & (
                treatment["time_to_progression"] == treatment["time_to_death"])]["time_to_progression"])
    III_total = III_III_count + III_IV_count + III_death_count

    # Stage IV transition counts
    stage_III_start = int((treatment[(treatment["stage_at_diagnosis"] == "III") & (
                treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_death"] - treatment[
                               (treatment["stage_at_diagnosis"] == "III") & (
                                           treatment["time_to_progression"] != treatment["time_to_death"])][
                               "time_to_progression"]).sum())
    IV_IV_count = int((treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"].apply(
        np.floor)).sum()) + stage_III_start  # Add with Stage III sum from progression time to death time
    IV_death_count = len(treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"]) + len(treatment[(
                                                                                                                            treatment[
                                                                                                                                "stage_at_diagnosis"] == "III") & (
                                                                                                                            treatment[
                                                                                                                                "time_to_progression"] !=
                                                                                                                            treatment[
                                                                                                                                "time_to_death"])])
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

# Simulate Microsimulation Output
time_horizon = 80  # months
states = ["III", "IV", "Dead"]
wtp_threshold = 100000
n_simulations = 1000

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


def simulate_all():
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
    paths_ctrl = generate_paths("B")
    costs_new, qalys_new = compute_costs_qalys(paths_new, "A")
    costs_ctrl, qalys_ctrl = compute_costs_qalys(paths_ctrl, "B")

    return paths_new, costs_new, qalys_new, paths_ctrl, costs_ctrl, qalys_ctrl


paths_new, costs_new, qalys_new, paths_ctrl, costs_ctrl, qalys_ctrl = simulate_all()

# ---------------- Streamlit UI ---------------- #
st.set_page_config(layout="wide")
st.title("Lung Cancer Trial Outcomes")

# Sidebar
st.sidebar.header("Filters")
plot_type = st.sidebar.selectbox("Select Plot", ["Survival Curves", "Cumulative Costs", "CEAC"])
age_filter = st.sidebar.slider("Select Age Range", 50, 80, (55, 65))
sex_filter = st.sidebar.selectbox("Select Sex", ["All", "Male", "Female"])
disease_stage = st.sidebar.selectbox("Select Stage", ["III", "IV"])

cost_range = st.sidebar.number_input("Drug A Cost", min_value=0, step=100)

# pricing_strategy = st.sidebar.selectbox("Pricing Strategy", ["CEA", "GRACE", "Dynamic Pricing", "Stacked Cohorts"])

# Apply filter
if sex_filter != "All":
    filtered_data = patient_data[patient_data["sex"] == sex_filter]
else:
    filtered_data = patient_data

# Main area plots
col1, col2 = st.columns([1, 3])
with col2:
    if plot_type == "Survival Curves":
        kmf = KaplanMeierFitter()
        fig = plt.figure(figsize=(8, 5))

        for treatment, data in zip(["A", "B"], [paths_new, paths_ctrl]):
            kmf.fit(durations=pd.Series([len(pd.Series(path)[pd.Series(path) != "Dead"]) for path in data]),
                    event_observed=[("Dead" in path) for path in data], label=treatment)
            kmf.plot_survival_function()

        plt.title("Survival Curves by Treatment")
        plt.xlabel("Months")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        st.pyplot(fig)

    elif plot_type == "Cumulative Costs":
        def cumulative_costs(paths, treatment):
            cum_costs = np.zeros((n_simulations, time_horizon))
            for i, path in enumerate(paths):
                for t in range(time_horizon):
                    cum_costs[i, t] = costs[treatment][path[t]] + (cum_costs[i, t - 1] if t > 0 else 0)
            return np.mean(cum_costs, axis=0)


        cum_costs_new = cumulative_costs(paths_new, "A")
        cum_costs_ctrl = cumulative_costs(paths_ctrl, "B")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cum_costs_new, label="Treatment A")
        ax.plot(cum_costs_ctrl, label="Treatment B")
        plt.xlabel("Month")
        plt.ylabel("Cumulative Cost ($)")
        plt.title("Cost Over Time")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig)

    elif plot_type == "CEAC":
        # Cost-Effectiveness Acceptability Curve (CEAC)
        wtp_values = np.arange(0, 200000, 1000)
        cost_mults = {"CEA": 1.0, "GRACE": 1.05, "Dynamic Pricing": 1.1, "Stacked Cohorts": 0.95}

        # Different WTP thresholds for different approaches
        probs_ceac = {}
        for label, cost_mult in cost_mults.items():
            adj_costs_new = costs_new * cost_mult
            probs = []
            for wtp in wtp_values:
                nmb_new = qalys_new * wtp - adj_costs_new
                nmb_ctrl = qalys_ctrl * wtp - costs_ctrl
                probs.append(np.mean(nmb_new > nmb_ctrl))
            probs_ceac[label] = probs

        plt.figure(figsize=(8, 5))
        for label, probs in probs_ceac.items():
            plt.plot(wtp_values, probs, label=label)
        plt.axvline(wtp_threshold, color="gray", linestyle="--", label="WTP Threshold")
        plt.xlabel("WTP per QALY ($)")
        plt.ylabel("Probability Cost-Effective")
        plt.title("Cost-Effectiveness Acceptability by Pricing Strategy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)