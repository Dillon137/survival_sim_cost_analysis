import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lifelines import KaplanMeierFitter

from patient_gen import generate_clinical_data, run_simulation

np.random.seed(42)

# ---------------- Streamlit UI ---------------- #
st.set_page_config(layout="wide")
st.title("Lung Cancer Trial Outcomes")

# Sidebar
st.sidebar.header("Filters")
plot_type = st.sidebar.selectbox("Select Plot", ["Survival Curves", "Cumulative Costs", "CEAC"])
age_filter = st.sidebar.slider("Select Age Range", 50, 80, (55, 65))
sex_filter = st.sidebar.selectbox("Select Sex (Not currently linked!)", ["All", "Male", "Female"])
disease_stage = st.sidebar.selectbox("Select Stage (Not currently linked!)", ["III", "IV"])

n_patients = st.sidebar.number_input("Number of patients in trial", min_value=100, step=10)
n_simulations = st.sidebar.number_input("Number of simulations", min_value=n_patients, step=10)
time_horizon = st.sidebar.number_input("Length of simulation (months)", min_value=12, step=1)

cost_range = st.sidebar.number_input("Drug A Cost (Not currently linked!)", min_value=0, step=100)




# TODO add adjustable variable for costs
#Temporary set costs
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


if sex_filter == "All":
    sex = ["Male", "Female"]
else:
    sex = sex_filter

patient_data = generate_clinical_data(age_filter, sex)

paths_new, costs_new, qalys_new, paths_ctrl, costs_ctrl, qalys_ctrl = run_simulation(patient_data, time_horizon, n_simulations)


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
        fig = plt.figure(figsize=(7, 4))

        for treatment, data in zip(["Experimental", "Control"], [paths_new, paths_ctrl]):
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


        cum_costs_new = cumulative_costs(paths_new, "Experimental")
        cum_costs_ctrl = cumulative_costs(paths_ctrl, "Control")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cum_costs_new, label="Experimental Treatment")
        ax.plot(cum_costs_ctrl, label="Control Treatment")
        plt.xlabel("Month")
        plt.ylabel("Cumulative Cost ($)")
        plt.title("Cost Over Time")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig)

    elif plot_type == "CEAC":
        # Cost-Effectiveness Acceptability Curve (CEAC)
        wtp_values = np.arange(0, 200000, 1000)
        wtp_threshold = 100000


        # Currently using naive estimates for each pricing method
        cost_mults = {"CEA": 1.0, "GRACE": 1.05, "Dynamic Pricing": 1.1, "Stacked Cohorts": 0.95}

        # Different WTP thresholds for different approaches
        probs_ceac = {}
        for label, cost_mult in cost_mults.items():
            adj_costs_new = costs_new * cost_mult
            probs = []
            for wtp in wtp_values:
                nmb_new = qalys_new*wtp - adj_costs_new
                nmb_ctrl = qalys_ctrl*wtp - costs_ctrl
                probs.append(np.mean(nmb_new > nmb_ctrl))
            probs_ceac[label] = probs

        plt.figure(figsize=(7, 4))
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










    # Debugging CEAC
    # mean_nmb_diff = [
    #     (np.mean(qalys_new) * wtp - np.mean(costs_new)) -
    #     (np.mean(qalys_ctrl) * wtp - np.mean(costs_ctrl))
    #     for wtp in wtp_values
    # ]
    #
    # # Plotting
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(wtp_values, mean_nmb_diff, marker='o', linestyle='-')
    # ax.set_title("Mean Net Monetary Benefit Difference vs Willingness to Pay")
    # ax.set_xlabel("Willingness to Pay (USD per QALY)")
    # ax.set_ylabel("Mean NMB Difference (Treatment Experimental - Control)")
    # ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    # ax.grid(True)
    #
    # # Show plot in Streamlit
    # st.pyplot(fig)