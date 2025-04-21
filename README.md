
# Lung Cancer Clinical Trial Generation and Cost-Effectiveness Simulator

This project simulates a cost-effectiveness analysis of two lung cancer treatments using synthetic patient data. Built using Python and Streamlit, the app generates synthetic clinical data, models disease progression via Markov microsimulation, and visualizes outcomes including survival curves, cumulative costs, and cost-effectiveness acceptability curves (CEAC).

## Features

- Simulates progression and survival for synthetic lung cancer patients.
- Estimates costs and QALYs using treatment-specific state transition models.
- Computes ICER and CEAC for various pricing strategies.
- Interactive filtering by patient characteristics (age, sex).
- Streamlit UI for dynamic visualizations.

---

## Project Structure

### 1. `patient_gen.py`

Contains core functions for patient data generation and simulation logic:

#### Functions:

- `generate_clinical_data(...)`
  - Creates a synthetic dataset of lung cancer patients.
  - Factors: age, sex, performance status, smoking status, diagnosis stage, treatment.
  - Survival modeled using Weibull distribution, adjusted for patient characteristics.

- `run_simulation(...)`
  - Converts patient-level outcomes into transition probabilities for a 3-state Markov model:
    - Stable Disease (Stage III)
    - Progressed Disease (Stage IV)
    - Death
  - Runs microsimulation for a defined time horizon and number of patients.
  - Outputs simulated patient state paths and computes total cost and QALY per simulation.

---

### 2. `app.py`

The Streamlit interface for user interaction.

#### Features:
- User selects:
  - Age range, sex, stage at diagnosis
  - Number of patients, time horizon, simulation count
  - Type of visualization
  

- Displays:
  - Survival curves
  - Costs over time
  - Cost-effectiveness acceptability curves (CEAC) 

---

## Visualizations
1. **Survival Curves**: Uses lifelines' Kaplan-Meier fitter on microsimulation results.
2. **Cumulative Costs**: Tracks mean cumulative costs over time for each treatment group.
3. **CEAC**: Plots cost-effectiveness probabilities across varying willingness-to-pay (WTP) thresholds for multiple pricing methods:
   - Standard CEA
   - GRACE pricing (5% markup)
   - Dynamic pricing (10% markup)
   - Stacked cohorts (-5% discount)

---

## Assumptions & Limitations

- **Survival Modeling**: Uses Weibull distributions with parameter adjustments based on patient demographics; real-world calibration is needed for clinical use.
- **Transition Probabilities**: Derived directly from TTP and TTD estimates; no external validation.
- **Cost Estimates**: Hardcoded per-month costs per state; user-defined input not yet fully integrated.
- **Filtering**: Some filters (e.g., disease stage, cost sliders) are not fully wired into backend logic.
- **Data**: All data is synthetic and not based on actual clinical datasets.

---

## Requirements
- `numpy`, `pandas`, `streamlit`, `lifelines`, `matplotlib`

---

## To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements
- Add sliders for cost inputs and dynamically bind them to simulation.
- Enable stage-specific filtering and visualization.
- Incorporate real-world clinical data or user-uploaded patient cohorts.
- Incorporate real-world financial data.
- Extend to multiple treatment arms or treatment sequences.
