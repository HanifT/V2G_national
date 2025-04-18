# %%
import warnings

warnings.filterwarnings("ignore")
import sys
import pickle

sys.path.append('D:\\Hanif\\V2G_national\\charging\\src')
import utilities
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("pyomo").setLevel(logging.WARNING)
# %% Input eVMT
if os.path.exists("D:\\Hanif\\V2G_national\\charging\\nhts_state_data.pkl"):
    with open("D:\\Hanif\\V2G_national\\charging\\nhts_state_data.pkl", "rb") as f:
        nstate_counts, state_abbrev_to_name = pickle.load(f)
# Define the electricity price file path
electricity_price_file = "D:\\Hanif\\V2G_national\\charging\\src\\weighted_hourly_prices.json"
# Directory to save pickle files
output_dir = "D:\\Hanif\\V2G_national\\results\\state_itineraries"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
# %% Define selected states (the ones you want to run first)
selected_states = [
    "DC",
    "NV",
    "AR",
    "MS", "WV",
    # "CT", "NM", "LA", "RI", "WY", "DE", "ND",
    # "KS", "NH", "KY", "NE", "ME", "AL", "MT", "SD", "ID", "UT", "VT", "OR",
    # "TN", "MA", "CO", "MO", "IN", "NJ", "WA", "MN", "VA", "MI", "IL", "PA",
    # "OH", "OK", "FL", "MD", "AZ", "IA", "SC",
    "GA", "NC", "WI",
    "NY",
    "CA",
    "TX"
]

# %%
# output_dir_base = "D:\\Hanif\\V2G_national\\results\\state_itineraries_nocost"  # normal Price
# itinerary_kwargs = {
#     # 'home_charger_likelihood': 75,
#     # 'work_charger_likelihood': 75,
#     # 'destination_charger_likelihood': 0.75,
#     'home_charger_power': 6.6,
#     'work_charger_power': 7.2,
#     'destination_charger_power': 100.1,
#     "ad_hoc_charger_power": 100.1,
#     'max_soc': 1,
#     'min_soc': 0.1,
# }
# utilities.run_simulation(selected_states_input=selected_states,
#                nstate_counts=nstate_counts,
#                electricity_price_file=electricity_price_file,
#                state_abbrev_to_name=state_abbrev_to_name,
#                output_dir=output_dir_base,
#                itinerary_kwargs=itinerary_kwargs,
#                r=1, c=1, o=1)

output_dir0 = "D:\\Hanif\\V2G_national\\results\\state_itineraries"  # normal Price
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir0,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir1 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapH"  # cheaper home charging
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir1,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=0.9, o=0.9)

output_dir2 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapW"  # cheaper work charging
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir2,
                         itinerary_kwargs=itinerary_kwargs,
                         r=0.9, c=1, o=0.9)

output_dir3 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapO"  # cheaper other charging
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir3,
                         itinerary_kwargs=itinerary_kwargs,
                         r=0.9, c=0.9, o=1)

output_dir4 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_6.6"  # normal price, 100 access H 6kw
itinerary_kwargs = {
    'home_charger_likelihood': 1,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir4,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir5 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_12"  # normal price, 100 access H 12 kw
itinerary_kwargs = {
    'home_charger_likelihood': 1,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 12,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir5,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir6 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_19"  # normal price, 100 access H 19 kw
itinerary_kwargs = {
    'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 19,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir6,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir7 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_6.6"  # normal price, 100 access W 6.6 kw
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    'work_charger_likelihood': 1,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir7,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir8 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_12"  # normal price, 100 access W 12 kw
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    'work_charger_likelihood': 75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 12,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir8,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir9 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_19"  # normal price, 100 access W 19 kw
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    'work_charger_likelihood': 1,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 19,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir9,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)
output_dir10 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_6.6"  # normal price, 75 access H W 6.6 kw
itinerary_kwargs = {
    'home_charger_likelihood': 0.75,
    'work_charger_likelihood': 0.75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir10,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir11 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_12"  # normal price, 75 access H W 12 kw
itinerary_kwargs = {
    'home_charger_likelihood': 0.75,
    'work_charger_likelihood': 0.75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 12,
    'work_charger_power': 12,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir11,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir12 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_19"  # normal price, 75  access H W 19 kw
itinerary_kwargs = {
    'home_charger_likelihood': 0.75,
    'work_charger_likelihood': 0.75,
    # 'destination_charger_likelihood': 0.75,
    'home_charger_power': 19,
    'work_charger_power': 19,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir12,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

output_dir13 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_25O"  # normal price, 25  access O
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    'destination_charger_likelihood': 0.25,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir13,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)
output_dir14 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_50O"  # normal price, 50  access O
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    'destination_charger_likelihood': 0.50,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir14,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)
output_dir15 = "D:\\Hanif\\V2G_national\\results\\state_itineraries_75O"  # normal price, 75  access O
itinerary_kwargs = {
    # 'home_charger_likelihood': 75,
    # 'work_charger_likelihood': 75,
    'destination_charger_likelihood': 0.75,
    'home_charger_power': 6.6,
    'work_charger_power': 7.2,
    'destination_charger_power': 100.1,
    "ad_hoc_charger_power": 100.1,
    'max_soc': 1,
    'min_soc': 0.1,
}
utilities.run_simulation(selected_states_input=selected_states,
                         nstate_counts=nstate_counts,
                         electricity_price_file=electricity_price_file,
                         state_abbrev_to_name=state_abbrev_to_name,
                         output_dir=output_dir15,
                         itinerary_kwargs=itinerary_kwargs,
                         r=1, c=1, o=1)

# %%

output_dirs = [
    # output_dir_base,
    output_dir0,
    output_dir1,
    # output_dir2,
    # output_dir3,
    # output_dir4, output_dir5, output_dir6,
    # output_dir7, output_dir8, output_dir9,
    # output_dir10, output_dir11, output_dir12,
    # output_dir13, output_dir14, output_dir15
]

for dir_path in output_dirs:
    utilities.plot_charging_distribution(state="DC", data_dir=dir_path)

# %%

state = "DC"
scenario_dirs = {
    # "Baseline scenarios": "D:\\Hanif\\V2G_national\\results\\state_itineraries_nocost",
    "Baseline scenarios": "D:\\Hanif\\V2G_national\\results\\state_itineraries",
    "Incentivized Home Charging (↑Work & Public Price)": "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapH",
    "Incentivized Workplace Charging (↑Home & Public Price)": "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapW",
    "Incentivized Public Charging (↑Home & Work Price)": "D:\\Hanif\\V2G_national\\results\\state_itineraries_cheapO",
    "Full Home Access–6.6 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_6.6",
    "Full Home Access–12 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_12",
    "Full Home Access–19 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100H_19",
    "Full Work Access–6.6 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_6.6",
    "Full Work Access–12 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_12",
    "Full Work Access–19 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100W_19",
    "High Access Home & Work–6.6 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_6.6",
    "High Access Home & Work–12 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_12",
    "High Access Home & Work–19 kW": "D:\\Hanif\\V2G_national\\results\\state_itineraries_100HW_19",
    "25% Public Charger Availability": "D:\\Hanif\\V2G_national\\results\\state_itineraries_25O",
    "50% Public Charger Availability": "D:\\Hanif\\V2G_national\\results\\state_itineraries_50O",
    "75% Public Charger Availability": "D:\\Hanif\\V2G_national\\results\\state_itineraries_75O"
}

full_df, cost_summary = utilities.load_full_charging(state, scenario_dirs)

# Step 1: Load weekly charging profiles
weekly_df = utilities.load_weekly_charging(state, scenario_dirs)

# Step 2: Normalize demand
normalized_df = utilities.normalize_weekly_demand(weekly_df)

# Step 3: Calculate flatness and variability
flex_df = utilities.calculate_flexibility(normalized_df)

# Step 4: Calculate off-peak charging share
off_peak_df = utilities.compute_off_peak_share(normalized_df)

# Step 5: Plot the flexibility matrix
utilities.plot_charging_flexibility(flex_df, off_peak_df, state)

entropy_df = utilities.compute_entropy_by_scenario(normalized_df)
utilities.plot_entropy_metrics_with_shapes(entropy_df, state)

solar_profile = np.array([
    0.00, 0.00, 0.00, 0.01, 0.03, 0.08, 0.20, 0.40, 0.65, 0.90, 1.00, 0.95,
    0.85, 0.70, 0.50, 0.35, 0.20, 0.10, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00
])

wind_profile = np.array([
    0.95, 1.00, 0.96, 0.94, 0.92, 0.88, 0.84, 0.80, 0.76, 0.74, 0.72, 0.70,
    0.72, 0.75, 0.78, 0.82, 0.87, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99
])

# solar_profile /= solar_profile.sum()

# wind_profile = np.array([0.8, 0.85, 0.90, 0.92, 0.96, 0.93, 0.88, 0.90, 0.87, 0.81, 0.71, 0.71,
#                          0.69, 0.81, 0.91, 0.92, 0.92, 0.92, 0.96, 1, 0.92, 0.95, 0.96, 0.94])
# wind_profile /= wind_profile.sum()

# Make sure you've defined normalized_df before this step
alignment_df = utilities.compute_alignment_scores(normalized_df, solar_profile, wind_profile)
utilities.plot_alignment_scores(alignment_df, state)

# Re-import necessary packages after state reset
import matplotlib.pyplot as plt

# Step 1: Determine peak in the baseline
baseline_peak = weekly_df[weekly_df["Scenario"] == "Baseline scenarios"]["Charging_kWh"].max()
threshold = 0.95 * baseline_peak

# Step 2: Identify high-peak hours per scenario
high_peak_df = weekly_df[
    (weekly_df["Charging_kWh"] >= threshold) &
    (weekly_df["Scenario"] != "Baseline scenarios")  # Exclude baseline
    ]
hourly_scenario_counts = high_peak_df.groupby(["HourOfWeek", "Scenario"]).size().unstack(fill_value=0)

# Step 3: Format x-axis labels
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hour_labels = [f"{h:02d}:00 {day_labels[d]}" for d in range(7) for h in range(24)]
hour_label_map = dict(zip(range(168), hour_labels))
hourly_scenario_counts.index = hourly_scenario_counts.index.map(hour_label_map)

# Step 4: Plot stacked bar chart
plt.figure(figsize=(16, 7))
ax = hourly_scenario_counts.plot(kind="bar", stacked=True, width=1.0, colormap="tab20", figsize=(16, 7), ax=plt.gca())

plt.title("High-Peak Charging Periods Across Scenarios (≥80% of Baseline Peak)", fontsize=15)
plt.xlabel("Hour of Week", fontsize=14)
plt.ylabel("Number of Scenarios with High-Peak Load", fontsize=14)
plt.xticks(rotation=90, fontsize=11)

# Force Y-axis to show integer values only
y_max = int(np.ceil(hourly_scenario_counts.sum(axis=1).max()))
plt.yticks(np.arange(0, y_max + 1, 1), fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Move legend to bottom
plt.legend(title="Scenario", loc='upper center', bbox_to_anchor=(0.5, -0.22),
           ncol=3, fontsize=11, title_fontsize=13, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
plt.show()
