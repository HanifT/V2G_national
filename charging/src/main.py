# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import pandas as pd
from utilities import load_itineraries, read_pkl
import optimization
import process
from reload import deep_reload
import numpy as np
from itineraries import MakeItineraries
from adoption import combine_csv_files
# %% Input eVMT

deep_reload('process')
# Reading the input files
codex_path = '/Users/haniftayarani/V2G_national/charging/codex.json'
verbose = True
data = process.load(codex_path)
data_cahrging = data["charging"]
data_trips = data["trips"]

data_cahrging_summary, data_cahrging_grouped = process.plot_days_between_charges(data_cahrging, min_energy=10, min_days=-1, max_days=15, excluded_makes=[])
charging_demand_curve = process.create_charging_demand_curve(data_cahrging_summary)
charging_demand_curve_agg = process.create_charging_demand_curve_agg(data_cahrging_summary)
process.plot_charging_demand_curve(charging_demand_curve)
process.plot_charging_demand_curve(charging_demand_curve_agg)
# %% Creating Itineraries

# MakeItineraries(day_type="all")
# MakeItineraries(day_type="weekday")
# MakeItineraries(day_type="weekend")

# %% NHTS mean
nhts17 = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub.csv")
nhts17 = nhts17[["HOUSEID", "PERSONID", "VEHID", "HHSTATE", "URBRUR", "HHSTFIPS", "VMT_MILE", "WTTRDFIN"]]

nhts17_grouped = nhts17.groupby(["HOUSEID", "PERSONID", "VEHID", "HHSTATE","URBRUR", "HHSTFIPS", "WTTRDFIN"])["VMT_MILE"].mean().reset_index(drop=False)
nhts17_grouped = nhts17_grouped[(nhts17_grouped["VEHID"] < 3) & (nhts17_grouped["VEHID"] > 0)]
nhts17_grouped = nhts17_grouped[nhts17_grouped["PERSONID"] < 3]
nhts17_grouped = nhts17_grouped[nhts17_grouped["VMT_MILE"] > 0]
nhts17_grouped_mean = nhts17_grouped.groupby(["HHSTATE", "HHSTFIPS", "URBRUR"]).apply(
    lambda x: pd.Series({"VMT_MILE_weighted_avg": (x["VMT_MILE"] * x["WTTRDFIN"]).sum() / x["WTTRDFIN"].sum()})
).reset_index()

state_abbrev_to_name = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
# Map the state names using HHSTFIPS
nhts17_grouped_mean["State"] = nhts17_grouped_mean["HHSTATE"].map(state_abbrev_to_name)
nhts17_grouped_mean_all = nhts17_grouped_mean.groupby(["State", "HHSTATE", "HHSTFIPS"])["VMT_MILE_weighted_avg"].mean().reset_index(drop=False)

# %%
combined_df = combine_csv_files("/Users/haniftayarani/V2G_national/charging/Data/Adoption/*.csv")
# %%


def generate_itineraries(day_types="all", num_itinerarie=3000, tile=14, states = ["WY"]):
    itineraries_total = load_itineraries(day_type=day_types)  # Change to False for weekend
    # Filter itineraries to include only those with HHSTATE == "WY"

    def filter_itineraries_by_states(itineraries, states):

        filtered_itineraries = [
            {"trips": itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)]}
            for itinerary in itineraries
            if not itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].empty
        ]
        return filtered_itineraries

    itineraries = filter_itineraries_by_states(itineraries_total, states)

    battery_capacities = [80 * 3.6e6, 55 * 3.6e6, 66 * 3.6e6, 65 * 3.6e6, 62 * 3.6e6,
                          75 * 3.6e6, 135 * 3.6e6, 60 * 3.6e6, 70 * 3.6e6, 80 * 3.6e6]
    probabilities = [0.42, 0.1, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.25]

    # Define a function to select a battery capacity based on the given probabilities
    def get_random_battery_capacity():
        return np.random.choice(battery_capacities, p=probabilities)

    # Other parameters for the itineraries
    itinerary_kwargs = {
        "tiles": tile,
        'initial_soc': 1,
        'final_soc': 1,
        'home_charger_likelihood': 0.95,
        'work_charger_likelihood': 0.60,
        "consumption": 782.928,
        'destination_charger_likelihood': 0.2,
        'home_charger_power': 6.1e3,
        'work_charger_power': 6.1e3,
        'destination_charger_power': 100.1e3,
        'max_soc': 1,
        'min_soc': .2,
        'min_dwell_event_duration': 15 * 60,
        'max_ad_hoc_event_duration': 7.2e3,
        'min_ad_hoc_event_duration': 5 * 60,
    }

    # Initialize an empty list to collect the results
    all_tailed_itineraries = []

    # Set the number of itineraries to loop through
    num_itineraries = num_itinerarie

    # Solver and itinerary parameters
    solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}

    # Loop over each itinerary in the list
    for n in range(num_itineraries):
        # Select a battery capacity based on the defined distribution
        selected_battery_capacity = get_random_battery_capacity()
        selected_energy_consumption = itineraries[n]["trips"]["Energy_Consumption"].mean()
        itinerary_kwargs['battery_capacity'] = selected_battery_capacity
        itinerary_kwargs['consumption'] = selected_energy_consumption
        # Instantiate the EVCSP class for each itinerary
        problem = optimization.EVCSP(itineraries[n], itinerary_kwargs=itinerary_kwargs)
        problem.Solve(solver_kwargs)

        # Print status and SIC for tracking
        print(f'Itinerary {n}: solver status - {problem.solver_status}, termination condition - {problem.solver_termination_condition}')
        print(f'Itinerary {n}: SIC - {problem.sic}')

        # Repeat (tail) the itinerary across tiles
        tiles = itinerary_kwargs["tiles"]
        tailed_itinerary = pd.concat([itineraries[n]['trips']] * tiles, ignore_index=True)

        # Add the SOC from the solution to the tailed itinerary
        soc_values = problem.solution['soc', 0].values  # Extract SOC values from problem.solution
        tailed_itinerary['SOC'] = soc_values[:len(tailed_itinerary)]

        # Add the SIC as a column to the tailed itinerary (repeated for each row)
        tailed_itinerary['SIC'] = problem.sic
        # Add the battery capacity as a column to the tailed itinerary
        tailed_itinerary['Battery Capacity'] = selected_battery_capacity  # <-- Add this line
        # tailed_itinerary['Battery Capacity'] = 60*3.6e6  # <-- Add this line

        # Append the tailed itinerary with SOC and SIC to the results list
        all_tailed_itineraries.append(tailed_itinerary)

    # Concatenate all the individual DataFrames into one final DataFrame
    final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
    final_df.name = f"final_df_{day_types}"
    return final_df
#%%

final_df_weekday = generate_itineraries(day_types="weekday", num_itinerarie=2000, tile=14, states=["CA"])
final_df_weekend = generate_itineraries(day_types="weekend", num_itinerarie=2000, tile=14, states=["CA"])
final_df_all = generate_itineraries(day_types="all", num_itinerarie=4000, tile=15, states=["CA"])
# %%


def post_processing(final_df, days=None):

    if days is None:
        days = list(range(1, 4))

    final_df = process.add_order_column(final_df)
    final_df_days = process.assign_days_of_week(final_df)
    final_df_days = process.identify_charging_sessions(final_df_days)
    final_df_days, final_df_days_grouped = process.calculate_days_between_charges_synt(final_df_days)
    final_df_days = process.map_whytrp1s_to_destination(final_df_days)
    final_df_days["Charging_Level"] = final_df_days.apply(process.determine_charging_level, axis=1)
    final_df_days["Charging_Speed"] = final_df_days["Charging_Level"].apply(process.determine_charging_speed)
    final_df_days["batt_kwh"] = final_df_days["Battery Capacity"].apply(process.batt_kwh)
    final_df_days = process.calculate_charging_times(final_df_days)
    final_df_days = process.calculate_trip_energy(final_df_days)
    final_df_days = process.calculate_charging_energy(final_df_days)
    final_df_days = process.assign_day_numbers(final_df_days)
    weekly_demand_curve = process.create_weekly_charging_demand(final_df_days)
    weekly_demand_curve = weekly_demand_curve[weekly_demand_curve["Day"].isin(days)]
    process.plot_charging_synt(weekly_demand_curve)
    return final_df_days, weekly_demand_curve


final_df_days_weekday, weekly_demand_curve_weekday = post_processing(final_df_weekday, list(range(8, 13)))
final_df_days_weekend, weekly_demand_curve_weekend = post_processing(final_df_weekend, list(range(6, 8)))
final_df_days_all, weekly_demand_curve_all = post_processing(final_df_all, list(range(8, 16)))

# %%


def calculate_actual_charging_end_time(df):
    # Filter rows with valid charging events
    charging_events = df[df["charging"] == True].copy()

    # Define a function to convert HHMM to minutes
    def hhmm_to_minutes(hhmm):
        hours = hhmm // 100
        minutes = hhmm % 100
        return hours * 60 + minutes

    # Define a function to convert minutes to HHMM
    def minutes_to_hhmm(minutes):
        hours = minutes // 60
        mins = minutes % 60
        return f"{int(hours):02d}{int(mins):02d}"

    hhmm_to_minutes(1445)
    hhmm_to_minutes(845)
    hhmm_to_minutes(1805)
    hhmm_to_minutes(2400)
    # Initialize the new column
    charging_events["Actual_Charging_End_Time"] = None

    # Iterate through each charging event
    for idx, row in charging_events.iterrows():
        # Charging start time in HHMM format
        start_time = int(row["Charging_Start_Time"])
        start_minutes = hhmm_to_minutes(start_time)

        # Calculate charging duration in minutes
        if row["Charging_Speed"] > 0:  # Prevent division by zero
            charging_duration_minutes = (row["Charged_Energy"] / row["Charging_Speed"]) * 60
        else:
            charging_duration_minutes = 0

        # Calculate the actual charging end time in total minutes
        actual_end_minutes = start_minutes + charging_duration_minutes

        # Convert the actual end time back to HHMM format
        actual_end_hhmm = minutes_to_hhmm(int(actual_end_minutes % 1440))  # Wrap around after 24 hours

        # Store the actual end time
        charging_events.loc[idx, "Actual_Charging_End_Time"] = actual_end_hhmm

        # Calculate the difference in minutes between Charging_End_Time and Actual_Charging_End_Time
        if pd.notna(row["Charging_End_Time"]):
            end_time = int(row["Charging_End_Time"])
            start_time = int(row["Charging_Start_Time"])
            end_minutes = hhmm_to_minutes(end_time)
            start_minutes = hhmm_to_minutes(start_time)

            # Adjust for cases where Actual End Time < Charging End Time (spanning midnight)
            if (actual_end_minutes < end_minutes) & (actual_end_minutes < start_minutes):
                diff_minutes = end_minutes - actual_end_minutes  # Done

            elif (actual_end_minutes < end_minutes) & (actual_end_minutes > start_minutes):
                diff_minutes = end_minutes - actual_end_minutes  # Done

            # elif (actual_end_minutes > end_minutes) & (actual_end_minutes < start_minutes):
            #     diff_minutes = (2400 - end_minutes) + actual_end_minutes  # Done

            elif (actual_end_minutes > end_minutes) & (actual_end_minutes > start_minutes):
                diff_minutes = (1440 - actual_end_minutes) + end_minutes  # Done

            charging_events.loc[idx, "ch_end_diff"] = minutes_to_hhmm(diff_minutes)
        else:
            charging_events.loc[idx, "ch_end_diff"] = None

    # Merge the updated data back into the original DataFrame
    df = df.merge(
        charging_events[["HOUSEID", "PERSONID", "VEHID", "order", "Actual_Charging_End_Time", "ch_end_diff"]],
        on=["HOUSEID", "PERSONID", "VEHID", "order"],
        how="left"
    )
    df["ch_end_diff"] = df["ch_end_diff"].fillna(0)
    df["ch_end_diff"] = df["ch_end_diff"].astype(int)
    df.loc[df["ch_end_diff"] < 0, "ch_end_diff"] = 0

    return df


# Apply the function to your DataFrame
final_df_days_charging = calculate_actual_charging_end_time(final_df_days)
final_df_days_charging = final_df_days_charging[final_df_days_charging["charging"] == True]

# %%
def hhmm_to_minutes(hhmm):
    hours = hhmm // 100
    minutes = hhmm % 100
    return hours * 60 + minutes

# Calculate charging end difference in minutes
final_df_days_charging["ch_end_diff_min"] = final_df_days_charging.apply(
    lambda row: hhmm_to_minutes(row["ch_end_diff"]),
    axis=1
)

import matplotlib.pyplot as plt
# Plot histogram
plt.figure(figsize=(8, 6))
plt.bar(
    final_df_days_charging["ch_end_diff_min"].value_counts().index,
    final_df_days_charging["ch_end_diff_min"].value_counts().values,
    width=5,
    edgecolor='black'
)
plt.title("Histogram of Charging End vs. Leaving Time Differences")
plt.xlabel("Difference in Minutes")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, float('inf')]
labels = [f"{bins[i]}-{bins[i+1]}" if bins[i+1] != float('inf') else f"{bins[i]}+" for i in range(len(bins)-1)]
final_df_days_charging["ch_end_diff_range"] = pd.cut(final_df_days_charging["ch_end_diff_min"], bins=bins, labels=labels, right=False)

# Calculate percentage of data in each range
range_counts = final_df_days_charging["ch_end_diff_range"].value_counts(normalize=True) * 100

# Sort the range_counts based on the bin order
range_counts = range_counts.reindex(labels)

# Plot the percentages as a bar chart with the correct order
plt.figure(figsize=(10, 6))
plt.bar(range_counts.index, range_counts.values, color='skyblue', edgecolor='black')
plt.title("Percentage of Data in Charging End Difference Ranges")
plt.xlabel("Difference Ranges (Minutes)")
plt.xticks(rotation=45)
plt.ylabel("Percentage (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
