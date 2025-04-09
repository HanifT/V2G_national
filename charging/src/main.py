# %%
import warnings
warnings.filterwarnings("ignore")
import sys
import pickle
sys.path.append('D:\\Hanif\\V2G_national\\charging\\src')
from utilities import run_simulation, plot_charging_distribution
import logging
import os

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
    # "DC",
    # "NV",
    # "AR",
    # "MS", "WV",
    # "CT", "NM", "LA", "RI", "WY", "DE", "ND",
    # "KS", "NH", "KY", "NE", "ME", "AL", "MT", "SD", "ID", "UT", "VT", "OR",
    # "TN", "MA", "CO", "MO", "IN", "NJ", "WA", "MN", "VA", "MI", "IL", "PA",
    # "OH", "OK", "FL", "MD", "AZ", "IA", "SC",
    # "GA",
    # "NC",
    # "WI", "NY",
    "CA",
    # "TX"
]

# %%
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
               nstate_counts=nstate_counts,
               electricity_price_file=electricity_price_file,
               state_abbrev_to_name=state_abbrev_to_name,
               output_dir=output_dir1,
               itinerary_kwargs=itinerary_kwargs,
               r=1, c=0.1, o=0.1)


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
run_simulation(selected_states_input=selected_states,
               nstate_counts=nstate_counts,
               electricity_price_file=electricity_price_file,
               state_abbrev_to_name=state_abbrev_to_name,
               output_dir=output_dir2,
               itinerary_kwargs=itinerary_kwargs,
               r=0.1, c=1, o=0.1)

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
run_simulation(selected_states_input=selected_states,
               nstate_counts=nstate_counts,
               electricity_price_file=electricity_price_file,
               state_abbrev_to_name=state_abbrev_to_name,
               output_dir=output_dir3,
               itinerary_kwargs=itinerary_kwargs,
               r=0.1, c=0.1, o=1)

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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
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
run_simulation(selected_states_input=selected_states,
               nstate_counts=nstate_counts,
               electricity_price_file=electricity_price_file,
               state_abbrev_to_name=state_abbrev_to_name,
               output_dir=output_dir15,
               itinerary_kwargs=itinerary_kwargs,
               r=1, c=1, o=1)



# %%


plot_charging_distribution(state="NV", data_dir=output_dir0)
plot_charging_distribution(state="NV", data_dir=output_dir1)
plot_charging_distribution(state="NV", data_dir=output_dir2)
plot_charging_distribution(state="NV", data_dir=output_dir3)
plot_charging_distribution(state="NV", data_dir=output_dir4)
plot_charging_distribution(state="NV", data_dir=output_dir5)
plot_charging_distribution(state="NV", data_dir=output_dir6)
plot_charging_distribution(state="NV", data_dir=output_dir7)
plot_charging_distribution(state="NV", data_dir=output_dir8)
plot_charging_distribution(state="NV", data_dir=output_dir9)
plot_charging_distribution(state="NV", data_dir=output_dir10)
plot_charging_distribution(state="NV", data_dir=output_dir11)
plot_charging_distribution(state="NV", data_dir=output_dir12)
plot_charging_distribution(state="NV", data_dir=output_dir13)
plot_charging_distribution(state="NV", data_dir=output_dir14)


# %%
import pandas as pd
import numpy as np
import seaborn as sns
# Define all scenario directories
scenario_dirs = {
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

# Choose a representative state for analysis
state = "AR"
weekly_charging_data = []

# Loop through all scenarios
for scenario, path in scenario_dirs.items():
    file_path = os.path.join(path, f"{state}_itineraries.pkl")
    if not os.path.exists(file_path):
        continue  # Skip if file doesn't exist
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        df = pd.concat(data.values(), ignore_index=True)
    else:
        df = data
    if "Charging_kwh_distribution" not in df.columns:
        continue
    df = df.dropna(subset=["Charging_kwh_distribution"])
    df["Charging_kwh_distribution"] = df["Charging_kwh_distribution"].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Aggregate weekly charging distribution
    weekly_charging = np.zeros(7 * 24)
    for _, row in df.iterrows():
        for (day, hour), charge in row["Charging_kwh_distribution"].items():
            if 0 <= day <= 6 and 0 <= hour <= 23:
                weekly_charging[day * 24 + hour] += charge
    for i, kwh in enumerate(weekly_charging):
        weekly_charging_data.append({
            "Scenario": scenario,
            "HourOfWeek": i,
            "Charging_kWh": kwh
        })

# Convert to DataFrame for analysis
weekly_charging_df = pd.DataFrame(weekly_charging_data)

summary = []
for scenario, group in weekly_charging_df.groupby("Scenario"):
    total_weekly_kwh = group["Charging_kWh"].sum()
    avg_daily_kwh = total_weekly_kwh / 7

    peak_row = group.loc[group["Charging_kWh"].idxmax()]
    peak_load_kw = peak_row["Charging_kWh"]
    peak_hour = int(peak_row["HourOfWeek"])
    peak_day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][peak_hour // 24]

    summary.append({
        "Scenario": scenario,
        # "Total Charging kwh": total_weekly_kwh,
        # "Avg Daily Charging (kWh)": round(avg_daily_kwh, 2),
        "Peak Load (kW)": round(peak_load_kw, 2),
        "Peak Hour": peak_hour,
        "Peak Day": peak_day
    })

summary_df = pd.DataFrame(summary)


normalized_df = weekly_charging_df.copy()
peak_lookup = summary_df.set_index("Scenario")["Peak Load (kW)"]
normalized_df["Normalized_Load"] = normalized_df.apply(
    lambda row: row["Charging_kWh"] / peak_lookup[row["Scenario"]],
    axis=1
)

from scipy.stats import variation  # Coefficient of variation

flexibility_metrics = []
for scenario, group in normalized_df.groupby("Scenario"):
    std_dev = group["Normalized_Load"].std()
    coeff_var = variation(group["Normalized_Load"])  # std / mean
    flexibility_metrics.append({
        "Scenario": scenario,
        "Normalized StdDev": round(std_dev, 4),
        "FlatnessIndex": round(coeff_var, 4)
    })

flex_df = pd.DataFrame(flexibility_metrics)

def is_peak_hour(hour):
    day = hour // 24
    hour_in_day = hour % 24
    return 17 <= hour_in_day <= 20  #  12–8 PM

normalized_df["IsPeakHour"] = normalized_df["HourOfWeek"].apply(is_peak_hour)

off_peak_stats = normalized_df.groupby("Scenario").apply(
    lambda df: 1 - df[df["IsPeakHour"]]["Charging_kWh"].sum() / df["Charging_kWh"].sum()
).reset_index(name="Off-Peak Share")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

# Merge Off-Peak Share with Flatness Index
index_df = flex_df.merge(off_peak_stats, on="Scenario")

# Calculate midpoints for quadrant lines
x_mid = (index_df["Off-Peak Share"].max() + index_df["Off-Peak Share"].min()) / 2
y_mid = (index_df["FlatnessIndex"].max() + index_df["FlatnessIndex"].min()) / 2
# Define group mapping for shape assignment
group_shapes = {
    "Baseline scenarios": "o",
    "Incentivized Charging": "s",
    "Full Home Access": "P",
    "Full Work Access": "X",
    "High Access Home & Work": "v",
    "Public Charger Availability": "*"
}

# Scenario name mapping to group
scenario_to_group = {
    "Baseline scenarios": "Baseline scenarios",

    "Incentivized Home Charging (↑Work & Public Price)": "Incentivized Charging",
    "Incentivized Workplace Charging (↑Home & Public Price)": "Incentivized Charging",
    "Incentivized Public Charging (↑Home & Work Price)": "Incentivized Charging",

    "Full Home Access–6.6 kW": "Full Home Access",
    "Full Home Access–12 kW": "Full Home Access",
    "Full Home Access–19 kW": "Full Home Access",

    "Full Work Access–6.6 kW": "Full Work Access",
    "Full Work Access–12 kW": "Full Work Access",
    "Full Work Access–19 kW": "Full Work Access",

    "High Access Home & Work–6.6 kW": "High Access Home & Work",
    "High Access Home & Work–12 kW": "High Access Home & Work",
    "High Access Home & Work–19 kW": "High Access Home & Work",

    "25% Public Charger Availability": "Public Charger Availability",
    "50% Public Charger Availability": "Public Charger Availability",
    "75% Public Charger Availability": "Public Charger Availability"
}

# Add group and shape columns
index_df["Group"] = index_df["Scenario"].map(scenario_to_group)
index_df["Shape"] = index_df["Group"].map(group_shapes)

# Plot setup
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20.colors

# Plot each point with correct shape and color
for i, row in index_df.iterrows():
    plt.scatter(
        row["Off-Peak Share"],
        row["FlatnessIndex"],
        color=colors[i % len(colors)],
        marker=row["Shape"],
        s=150,
        label=row["Scenario"]
    )

# Add quadrant shading
plt.gca().add_patch(Rectangle((min(index_df["Off-Peak Share"]), y_mid),
                              x_mid - min(index_df["Off-Peak Share"]),
                              max(index_df["FlatnessIndex"]) - y_mid,
                              color='red', alpha=0.1))

plt.gca().add_patch(Rectangle((x_mid, min(index_df["FlatnessIndex"])),
                              max(index_df["Off-Peak Share"]) - x_mid,
                              y_mid - min(index_df["FlatnessIndex"]),
                              color='green', alpha=0.1))

# Quadrant lines
plt.axvline(x=x_mid, color='gray', linestyle='--')
plt.axhline(y=y_mid, color='gray', linestyle='--')

# Build custom legend
# Build custom legend
legend_handles = []
for i, row in index_df.iterrows():
    handle = mlines.Line2D(
        [], [],
        color=colors[i % len(colors)],
        marker=row["Shape"],
        linestyle='None',
        markersize=10,
        label=row["Scenario"]
    )
    legend_handles.append(handle)

# Add the legend at the bottom
plt.legend(
    handles=legend_handles,
    title="Scenario",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=12,
    title_fontsize=12,
    frameon=False
)

# Labels and layout
plt.xlabel("Off-Peak Charging Share (Higher = More Shifted)", fontsize=12)
plt.ylabel("Flatness Index (Lower = Flatter)", fontsize=12)
plt.title("Charging Flexibility Matrix")
plt.grid(True, linestyle='--', alpha=0.6)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.show()