import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter

# Load data
patient_data = pd.read_csv("./data/synthetic_lung_cancer_trial_data.csv")

# Initialize Markov states
states = ['III', 'IV', 'Death']
transition_counts = pd.DataFrame(0, index=states, columns=states)

treatment_A = patient_data[patient_data["treatment"] == "A"]
treatment_B = patient_data[patient_data["treatment"] == "B"]

probs = []
for treatment in [treatment_A, treatment_B]:
    treatment_probs = []
    
    # Stage III transition counts
    III_III_count = int((treatment[(treatment["stage_at_diagnosis"] == "III")]["time_to_progression"].apply(np.floor)).sum())
    III_IV_count = len(treatment[(treatment["stage_at_diagnosis"] == "III") & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_progression"])
    III_death_count = len(treatment[(treatment["stage_at_diagnosis"] == "III") & (treatment["time_to_progression"] == treatment["time_to_death"])]["time_to_progression"])
    III_total = III_III_count + III_IV_count + III_death_count

    # Stage IV transition counts
    stage_III_start = int((treatment[(treatment["stage_at_diagnosis"] == "III") & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_death"] - treatment[(treatment["stage_at_diagnosis"] == "III") & (treatment["time_to_progression"] != treatment["time_to_death"])]["time_to_progression"]).sum())
    IV_IV_count = int((treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"].apply(np.floor)).sum()) + stage_III_start # Add with Stage III sum from progression time to death time
    IV_death_count = len(treatment[(treatment["stage_at_diagnosis"] == "IV")]["time_to_death"]) + len(treatment[(treatment["stage_at_diagnosis"] == "III") & (treatment["time_to_progression"] != treatment["time_to_death"])])
    IV_total = IV_IV_count + IV_death_count

    # Stage III transition probabilities
    III_III_prob = III_III_count/III_total
    III_IV_prob = III_IV_count/III_total
    III_death_prob = III_death_count/III_total
    treatment_probs.append([III_III_prob, III_IV_prob, III_death_prob])
    
    # Stage IV transition probabilities
    IV_IV_prob = IV_IV_count/IV_total
    IV_death_prob = IV_death_count/IV_total
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

# transition_probs.to_csv("transition_matrix.csv")


# Kaplan-Meier Estimation
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))
for treatment in ["A", "B"]:
    mask = patient_data["treatment"] == treatment
    kmf.fit(durations=patient_data[mask]["time_to_death"], event_observed=None, label=treatment)
    kmf.plot_survival_function()

plt.title("Survival Curves by Treatment")
plt.xlabel("Months")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()



# Simulate Microsimulation Output
np.random.seed(42)

time_horizon = 80  # months
states = ["III", "IV", "Dead"]

# Define health state costs per month (USD)
costs = {
    "A": {
        "III": 300,
        "IV": 600,
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


# Generate dummy patient paths for two treatments
def generate_paths(treatment, n_simulations=500):
    paths = []
    
    for patient in range(n_patients):
        path = []
        state = np.random.choice(["III", "IV"], p=[0.3, 0.7])

        for t in range(time_horizon):
            base_probs = transition_probs[treatment][state]
            adjusted_probs = np.array(base_probs) + 1e-3
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            # print(adjusted_probs)
            
            alpha = np.array(adjusted_probs)*100
            # print(alpha)
            
            sampled_probs = np.random.dirichlet(alpha)
            # print(sampled_probs)
            # print()
            
            # state = np.random.choice(["III", "IV", "Dead"], p=sampled_probs)
            
            state = np.random.choice(["III", "IV", "Dead"], p=transition_probs[treatment][state])
            path.append(state)
            
            if state == "Dead":
                path.extend(["Dead"] * (time_horizon - len(path)))
                break
                
        paths.append(path)
        
    return paths

n_simulations=500
paths_A = generate_paths("A")
paths_B = generate_paths("B")







# Kaplan-Meier Estimation
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))
for treatment, data in zip(["A", "B"], [paths_A, paths_B]):
    kmf.fit(durations=pd.Series([len(pd.Series(path)[pd.Series(path) != "Dead"]) for path in data]), event_observed=[("Dead" in path) for path in data], label=treatment)
    kmf.plot_survival_function()

plt.title("Survival Curves by Treatment")
plt.xlabel("Months")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()














# Compute Costs and QALYs
def compute_costs_qalys(paths, treatment):
    total_costs = []
    total_qalys = []
    
    for path in paths:
        cost = sum([costs[treatment][state] for state in path])
        qaly = sum([qaly_weights[state] for state in path])/12  #Convert to annual QALYs
        total_costs.append(cost)
        total_qalys.append(qaly)
        
    return np.array(total_costs), np.array(total_qalys)


costs_ctrl, qalys_ctrl = compute_costs_qalys(paths_B, "B")
costs_new, qalys_new = compute_costs_qalys(paths_A, "A")

costs_ctrl = costs_ctrl[:min(len(costs_ctrl), len(costs_new))]
qalys_ctrl = qalys_ctrl[:min(len(qalys_ctrl), len(qalys_new))]

costs_new = costs_new[:min(len(costs_ctrl), len(costs_new))]
qalys_new = qalys_new[:min(len(qalys_ctrl), len(qalys_new))]

# ICER & Value-Based Price
delta_cost = np.mean(costs_new - costs_ctrl)
delta_qaly = np.mean(qalys_new - qalys_ctrl)
icer = delta_cost/delta_qaly

print(f"ICER (New vs Control): ${icer:,.2f} per QALY")

# Value-based price estimation
wtp_threshold = 100000
vbp = wtp_threshold * delta_qaly + np.mean(costs_ctrl)
print(f"Value-Based Price for New Treatment: ${vbp:,.2f}")



# Pricing approaches
vbp_cea = vbp
vbp_grace = vbp * 0.95  # simulate adjustment for risk
vbp_dynamic = vbp * 0.9  # simulate launch discount
vbp_stacked = vbp * 1.05  # simulate pricing for broader cohorts

# Bar chart for VBP
approaches = ["CEA", "GRACE", "Dynamic Pricing", "Stacked Cohorts"]
values = [vbp_cea, vbp_grace, vbp_dynamic, vbp_stacked]

plt.figure(figsize=(8, 5))
plt.bar(approaches, values, color=["skyblue", "lightgreen", "orange", "lightcoral"])
plt.ylabel("Value-Based Price ($)")
plt.title("Value-Based Price by Pricing Approach")
plt.xticks(rotation=15)
plt.tight_layout()
plt.grid(True, axis="y")
plt.show()

# Cost-Effectiveness Acceptability Curve (CEAC)
wtp_values = np.arange(0, 200000, 1000)
# probs_cost_effective_new = []
# probs_cost_effective_ctrl = []

# for wtp in wtp_values:
#     nmb_new = qalys_new * wtp - costs_new
#     nmb_ctrl = qalys_ctrl * wtp - costs_ctrl
#     probs_cost_effective_new.append(np.mean(nmb_new > nmb_ctrl))
#     probs_cost_effective_ctrl.append(np.mean(nmb_ctrl > nmb_new))


# Different WTP thresholds for different approaches
probs_ceac = {}
for label, cost_mult in zip(approaches, [1, 1.05, 1.1, 0.95]):
    adj_costs_new = costs_new * cost_mult
    probs = []
    for wtp in wtp_values:
        nmb_new = qalys_new * wtp - adj_costs_new
        nmb_ctrl = qalys_ctrl * wtp - costs_ctrl
        probs.append(np.mean(nmb_new > nmb_ctrl))
    probs_ceac[label] = probs

plt.figure(figsize=(9, 6))
for label, probs in probs_ceac.items():
    plt.plot(wtp_values, probs, label=label)
plt.axvline(wtp_threshold, color="gray", linestyle="--", label="WTP Threshold")
plt.xlabel("WTP per QALY ($)")
plt.ylabel("Probability Cost-Effective")
plt.title("Cost-Effectiveness Acceptability by Pricing Strategy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Cost Over Time Visualization
def cumulative_costs(paths, treatment):
    cum_costs = np.zeros((n_simulations, time_horizon))
    for i, path in enumerate(paths):
        for t in range(time_horizon):
            cum_costs[i, t] = costs[treatment][path[t]] + (cum_costs[i, t-1] if t > 0 else 0)
    return np.mean(cum_costs, axis=0)

avg_costs_ctrl = cumulative_costs(paths_B, "B")
avg_costs_new = cumulative_costs(paths_A, "A")

plt.figure(figsize=(8, 5))
plt.plot(avg_costs_ctrl, label="Control")
plt.plot(avg_costs_new, label="New Treatment")
plt.xlabel("Month")
plt.ylabel("Cumulative Cost ($)")
plt.title("Average Cost Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()