import sys

sys.path.append('D:\\Hanif\\V2G_national\\charging\\src')
import time
import pickle as pkl
from tqdm import tqdm
from temp import final_temp_adjustment
import process
import random
import calendar
import json
import pandas as pd
import os
import pickle
import scipy.stats as stats
import numpy as np
from collections import defaultdict
import time
# %%


def load_ev_registration_data():
    """Loads the EV registration data from the Excel file specified in codex.json."""
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the JSON file path based on the class logic
    config_path = os.path.join(current_dir, 'charging', 'src', 'codex.json')

    # Load the JSON file
    with open(config_path, "r") as f:
        data = json.load(f)

    # Define the data directory path based on the class logic
    data_dir = os.path.join(current_dir, 'charging', 'Data', 'AFDC')

    # Extract the Excel file details
    file_name, sheet_name = None, None
    for file in data["files"]:
        if file["name"] == "EV_register.xlsx":
            file_name = file["name"]
            sheet_name = file["sheet"]

    # Construct the full path dynamically
    file_path = os.path.join(data_dir, file_name)

    # Check if the file exists before reading
    if os.path.exists(file_path):
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        print(f"Error: File not found at {file_path}")
        return None


def load_nhts_state_data(pickle_path):
    """Loads state-level sample count data and replaces abbreviations with full state names."""
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            nstate_counts, state_abbrev_to_name = pickle.load(f)

        # Replace state abbreviations with full state names
        nstate_counts["HHSTATE"] = nstate_counts["HHSTATE"].map(state_abbrev_to_name)
        nstate_counts = nstate_counts.rename(columns={"count": "sample"})
        return nstate_counts, state_abbrev_to_name
    else:
        print(f"Error: File not found at {pickle_path}")
        return None


def merge_vehicle_data(registration_data, nstate_data):
    """Merges the EV registration data with state-level sample count data."""
    if registration_data is None or nstate_data is None:
        print("Error: One or more required datasets are missing.")
        return None

    vehicle_data = pd.merge(registration_data, nstate_data, left_on="State", right_on="HHSTATE", how="left")
    vehicle_data = vehicle_data.drop(columns="HHSTATE")
    vehicle_data = vehicle_data.iloc[:-1]  # Remove last row
    return vehicle_data


def calculate_smooth_sample_size(N, z, p):
    """Calculates required sample size using a smooth function for margin of error."""
    base_N = 500000  # Reference population size for base MoE
    base_moe = 0.02  # Max margin of error for small populations

    # Compute MoE using a power function to scale smoothly
    moe = base_moe * (base_N / N) ** 0.1  # Exponent controls decay rate

    # Compute initial sample size without finite population correction
    n = (z ** 2 * p * (1 - p)) / (moe ** 2)

    # Apply finite population correction
    n_adj = n / (1 + ((n - 1) / N))

    return np.ceil(n_adj)  # Round up


def compute_sample_requirements(vehicle_data, confidence_level=0.95):
    """Computes required sample size and additional samples needed."""
    if vehicle_data is None:
        print("Error: Vehicle data is missing.")
        return None

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)  # Z-score for CI
    p = 0.5  # Assumed proportion

    # Compute required sample size dynamically
    vehicle_data["Required Sample Size"] = vehicle_data["Registration Count"].apply(
        lambda N: calculate_smooth_sample_size(N, z_score, p)
    )

    # Compute additional samples needed
    vehicle_data["Additional Samples Needed"] = vehicle_data["Required Sample Size"] - vehicle_data["sample"]
    vehicle_data["Additional Samples Needed"] = vehicle_data["Additional Samples Needed"].apply(lambda x: max(x, 0))

    return vehicle_data


def tail_itineraries(df, tail):
    # Example: If your data is already grouped by household (or unique identifier),
    # you might use:
    return df.groupby(['HOUSEID', 'VEHID', 'PERSONID'], group_keys=False).apply(lambda x: x.tail(tail))


def MakeItineraries(day_type="all"):
    # Start timer
    t0 = time.time()
    registration = load_ev_registration_data()
    nstate_counts, state_abbrev_to_name = load_nhts_state_data("D:\\Hanif\\V2G_national\\charging\\nhts_state_data.pkl")
    vehicle_data = merge_vehicle_data(registration, nstate_counts)
    vehicle_data = compute_sample_requirements(vehicle_data)
    print('Loading NHTS data:', end='')

    # Load data
    trips = pd.read_csv('D:\\Hanif\\V2G_national\\charging\\Data\\NHTS_2017\\trippub.csv')
    temp = final_temp_adjustment()
    trips = pd.merge(trips, temp[["Energy_Consumption", "HHSTATE", "HHSTFIPS"]], on=["HHSTATE", "HHSTFIPS"], how="left")
    # Drop unnecessary columns
    trips = trips[['HOUSEID', 'PERSONID', 'TDTRPNUM', 'STRTTIME', 'ENDTIME', 'TRVLCMIN',
                   'TRPMILES', 'TRPTRANS', 'VEHID', 'WHYFROM', 'LOOP_TRIP', 'TRPHHVEH', "WHODROVE",
                   'PUBTRANS', 'TRIPPURP', 'DWELTIME', 'TDWKND', 'VMT_MILE', 'WHYTRP1S',
                   'TDCASEID', 'WHYTO', 'TRAVDAY', 'HOMEOWN', 'HHSIZE', 'HHVEHCNT',
                   'HHFAMINC', 'HHSTATE', 'HHSTFIPS', 'TDAYDATE', 'URBAN', 'URBANSIZE',
                   'URBRUR', 'GASPRICE', 'CENSUS_D', 'CENSUS_R', 'CDIVMSAR', 'VEHTYPE', 'WTTRDFIN',  "Energy_Consumption"]]
    # Create the mapping dictionary for HHFAMINC
    income_mapping = {
        -7: None,  # I prefer not to answer
        -8: None,  # I don't know
        -9: None,  # Not ascertained
        1: 5000,  # Less than $10,000
        2: 12500,  # $10,000 to $14,999
        3: 20000,  # $15,000 to $24,999
        4: 30000,  # $25,000 to $34,999
        5: 42500,  # $35,000 to $49,999
        6: 62500,  # $50,000 to $74,999
        7: 87500,  # $75,000 to $99,999
        8: 112500,  # $100,000 to $124,999
        9: 137500,  # $125,000 to $149,999
        10: 175000,  # $150,000 to $199,999
        11: 200000  # $200,000 or more (using 200k as a conservative estimate)
    }
    # Ensure HHFAMINC is numeric first
    trips['HHFAMINC'] = pd.to_numeric(trips['HHFAMINC'], errors='coerce')

    # Map the values to income
    trips['Income'] = trips['HHFAMINC'].map(income_mapping)
    # Exclude rows with negative HHFAMINC for the calculation
    state_avg_income = (
        trips[trips['HHFAMINC'] > 0]  # Exclude negative values
        .groupby('HHSTATE')['Income']
        .mean()
        .to_dict()
    )

    # Step 3: Replace Negative Incomes with State Average
    def fill_negative_income(row):
        if pd.isna(row['Income']):  # If income is NaN (negative or unknown)
            state = row['HHSTATE']
            return state_avg_income.get(state, 0)  # Use state average or 0 if state is missing
        return row['Income']

    # Apply the function to the dataframe
    trips['Income'] = trips.apply(fill_negative_income, axis=1)

    # Filter trips based on day type
    if day_type == "weekday":
        trips = trips[trips["TDWKND"] == 2]  # Select only weekdays
    elif day_type == "weekend":
        trips = trips[trips["TDWKND"] == 1]  # Select only weekends
    elif day_type == "all":
        pass  # No filtering needed for all days
    else:
        raise ValueError("day_type must be one of 'weekday', 'weekend', or 'all'")

    # Further filtering of trips based on trip and vehicle criteria
    trip_veh = [3, 4, 5, 6]
    veh_type = [-1, 1, 2, 3, 4, 5]

    trips = trips[trips["TRPTRANS"].isin(trip_veh)]
    trips = trips[trips["TRPMILES"] > 0]
    trips = trips[trips["TRVLCMIN"] > 0]
    trips = trips[trips["VEHTYPE"].isin(veh_type)]
    trips = trips[trips["TRPMILES"] < 100]
    trips = trips[trips['PERSONID'] == trips['WHODROVE']]
    # Selecting for household vehicles
    trips = trips[(
        (trips['VEHID'] <= 2)
    )].copy()
    new_itineraries = trips
    print(' {:.4f} seconds'.format(time.time() - t0))
    t0 = time.time()
    print('Creating itinerary dicts:', end='')
    # Selecting for household vehicles
    new_itineraries = new_itineraries[(new_itineraries['TRPHHVEH'] == 1)].copy()
    # Get unique combinations of household, vehicle, and person IDs
    unique_combinations = new_itineraries[['HOUSEID', 'VEHID', 'PERSONID']].drop_duplicates()
    # ** Generate Random Dates for Unique Itineraries Outside Loop **
    print('Generating random dates for unique itineraries:', end='')
    # Randomly select month and day for each unique itinerary
    random_months = np.random.randint(1, 13, len(unique_combinations))
    random_days = np.array([
        random.randint(1, calendar.monthrange(2017, month)[1]) for month in random_months
    ])
    unique_combinations['Month'] = random_months
    unique_combinations['Day'] = random_days
    unique_combinations['Year'] = 2017
    print(' Done.')
    # Merge back to new_itineraries
    new_itineraries = pd.merge(new_itineraries, unique_combinations, on=['HOUSEID', 'VEHID', 'PERSONID'], how='left')
    # Initialize an array to store each itinerary dictionary
    itineraries = np.array([None] * unique_combinations.shape[0])

    unique_combinations = new_itineraries[['HOUSEID', 'VEHID', 'PERSONID']].drop_duplicates()
    # Main loop: iterate over each unique household-vehicle-person combination in the test set
    for idx, row in tqdm(enumerate(unique_combinations.itertuples(index=False)), total=unique_combinations.shape[0]):
        hh_id = row.HOUSEID
        veh_id = row.VEHID
        person_id = row.PERSONID

        # Get trips for this specific household-vehicle-person combination
        trips_indices = np.argwhere(
            (new_itineraries['HOUSEID'] == hh_id) &
            (new_itineraries['VEHID'] == veh_id) &
            (new_itineraries['PERSONID'] == person_id)
        ).flatten()

        # Create the dictionary for the current household-vehicle-person combination
        itinerary_dict = {
            'trips': new_itineraries.iloc[trips_indices]
        }

        # Store in the array at the current index
        itineraries[idx] = itinerary_dict

    # Display time taken
    print('Itineraries creation took {:.4f} seconds'.format(time.time() - t0))

    # Set output file name based on day_type parameter
    if day_type == "weekday":
        output_file = 'D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data\\itineraries_weekday.pkl'
    elif day_type == "weekend":
        output_file = 'D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data\\itineraries_weekend.pkl'
    elif day_type == "all":
        output_file = 'D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data\\itineraries_all_days.pkl'

    # Save itineraries to a pickle file
    t0 = time.time()
    print('Pickling outputs:', end='')
    pkl.dump(itineraries, open(output_file, 'wb'))
    print(' Done in {:.4f} seconds'.format(time.time() - t0))


def MakeItineraries_bootstrap(confidence=0.95, output_direction="D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data"):
    # Start timer
    day_type = "all"
    t0 = time.time()
    registration = load_ev_registration_data()
    nstate_counts, state_abbrev_to_name = load_nhts_state_data("D:\\Hanif\\V2G_national\\charging\\nhts_state_data.pkl")
    vehicle_data = merge_vehicle_data(registration, nstate_counts)
    vehicle_data = compute_sample_requirements(vehicle_data, confidence_level=confidence)
    print('Loading NHTS data:', end='')
    # Load data
    trips = pd.read_csv('D:\\Hanif\\V2G_national\\charging\\Data\\NHTS_2017\\trippub.csv')
    temp = final_temp_adjustment()
    trips = pd.merge(trips, temp[["Energy_Consumption", "HHSTATE", "HHSTFIPS"]], on=["HHSTATE", "HHSTFIPS"], how="left")
    # Drop unnecessary columns
    trips = trips[['HOUSEID', 'PERSONID', 'TDTRPNUM', 'STRTTIME', 'ENDTIME', 'TRVLCMIN',
                   'TRPMILES', 'TRPTRANS', 'VEHID', 'WHYFROM', 'LOOP_TRIP', 'TRPHHVEH', "WHODROVE",
                   'PUBTRANS', 'TRIPPURP', 'DWELTIME', 'TDWKND', 'VMT_MILE', 'WHYTRP1S',
                   'TDCASEID', 'WHYTO', 'TRAVDAY', 'HOMEOWN', 'HHSIZE', 'HHVEHCNT',
                   'HHFAMINC', 'HHSTATE', 'HHSTFIPS', 'TDAYDATE', 'URBAN', 'URBANSIZE',
                   'URBRUR', 'GASPRICE', 'CENSUS_D', 'CENSUS_R', 'CDIVMSAR', 'VEHTYPE', 'WTTRDFIN',  "Energy_Consumption"]]
    # Create the mapping dictionary for HHFAMINC
    income_mapping = {
        -7: None,  # I prefer not to answer
        -8: None,  # I don't know
        -9: None,  # Not ascertained
        1: 5000,  # Less than $10,000
        2: 12500,  # $10,000 to $14,999
        3: 20000,  # $15,000 to $24,999
        4: 30000,  # $25,000 to $34,999
        5: 42500,  # $35,000 to $49,999
        6: 62500,  # $50,000 to $74,999
        7: 87500,  # $75,000 to $99,999
        8: 112500,  # $100,000 to $124,999
        9: 137500,  # $125,000 to $149,999
        10: 175000,  # $150,000 to $199,999
        11: 200000  # $200,000 or more (using 200k as a conservative estimate)
    }
    # Ensure HHFAMINC is numeric first
    trips['HHFAMINC'] = pd.to_numeric(trips['HHFAMINC'], errors='coerce')

    # Map the values to income
    trips['Income'] = trips['HHFAMINC'].map(income_mapping)
    # Exclude rows with negative HHFAMINC for the calculation
    state_avg_income = (
        trips[trips['HHFAMINC'] > 0]  # Exclude negative values
        .groupby('HHSTATE')['Income']
        .mean()
        .to_dict()
    )

    # Step 3: Replace Negative Incomes with State Average
    def fill_negative_income(row):
        if pd.isna(row['Income']):  # If income is NaN (negative or unknown)
            state = row['HHSTATE']
            return state_avg_income.get(state, 0)  # Use state average or 0 if state is missing
        return row['Income']

    # Apply the function to the dataframe
    trips['Income'] = trips.apply(fill_negative_income, axis=1)

    # Filter trips based on day type
    if day_type == "weekday":
        trips = trips[trips["TDWKND"] == 2]  # Select only weekdays
    elif day_type == "weekend":
        trips = trips[trips["TDWKND"] == 1]  # Select only weekends
    elif day_type == "all":
        pass  # No filtering needed for all days
    else:
        raise ValueError("day_type must be one of 'weekday', 'weekend', or 'all'")

    # Further filtering of trips based on trip and vehicle criteria
    trip_veh = [3, 4, 5, 6]
    veh_type = [-1, 1, 2, 3, 4, 5]

    trips = trips[trips["TRPTRANS"].isin(trip_veh)]
    trips = trips[trips["TRPMILES"] > 0]
    trips = trips[trips["TRVLCMIN"] > 0]
    trips = trips[trips["VEHTYPE"].isin(veh_type)]
    trips = trips[trips["TRPMILES"] < 100]
    trips = trips[trips['PERSONID'] == trips['WHODROVE']]
    # Selecting for household vehicles
    trips = trips[(
        (trips['VEHID'] <= 2)
    )].copy()
    new_itineraries = trips
    print(' {:.4f} seconds'.format(time.time() - t0))
    t0 = time.time()
    print('Creating itinerary dicts:', end='')
    # Selecting for household vehicles
    new_itineraries = new_itineraries[(new_itineraries['TRPHHVEH'] == 1)].copy()
    # Get unique combinations of household, vehicle, and person IDs
    unique_combinations = new_itineraries[['HOUSEID', 'VEHID', 'PERSONID']].drop_duplicates()
    # ** Generate Random Dates for Unique Itineraries Outside Loop **
    print('Generating random dates for unique itineraries:', end='')
    # Randomly select month and day for each unique itinerary
    random_months = np.random.randint(1, 13, len(unique_combinations))
    random_days = np.array([
        random.randint(1, calendar.monthrange(2017, month)[1]) for month in random_months
    ])
    unique_combinations['Month'] = random_months
    unique_combinations['Day'] = random_days
    unique_combinations['Year'] = 2017
    print(' Done.')
    # Merge back to new_itineraries
    new_itineraries = pd.merge(new_itineraries, unique_combinations, on=['HOUSEID', 'VEHID', 'PERSONID'], how='left')

    def bootstrap_and_create_7_day_itineraries_fixed(new_itineraries, vehicle_data, state_abbrev_to_name, weight_column="WTTRDFIN"):
        # Convert HHSTATE abbreviations to full state names
        new_itineraries["State_Full"] = new_itineraries["HHSTATE"].map(state_abbrev_to_name)

        # Filter to states that need bootstrapping
        states_to_bootstrap = vehicle_data[vehicle_data["Additional Samples Needed"] > 0]

        final_itineraries = []
        nhouseid_counter = 1  # To create unique NHOUSEID labels

        # Helper function: normalize weights
        def normalize_weights(df, column):
            total_weight = df[column].sum()
            return df[column] / total_weight if total_weight > 0 else np.ones(len(df)) / len(df)

        # Helper function: sample itinerary groups and return full trips
        def sample_itinerary_groups(unique_ids, full_trips, n_samples, weights):
            sampled_groups = unique_ids.sample(n=n_samples, replace=True, weights=weights).reset_index(drop=True)
            sampled_groups["BOOT_ID"] = sampled_groups.index.astype(str)
            sampled_trips = sampled_groups.merge(full_trips, on=["HOUSEID", "VEHID", "PERSONID"], how="left")
            sampled_trips["BOOT_ID"] = sampled_trips["BOOT_ID"].astype(str)
            return sampled_trips

        for _, row in states_to_bootstrap.iterrows():
            state_name = row["State"]
            required_n = int(row["Required Sample Size"])

            state_trips = new_itineraries[new_itineraries["State_Full"] == state_name]
            if state_trips.empty:
                print(f"Warning: No trip data for {state_name}. Skipping...")
                continue

            # Separate weekday & weekend trips
            weekday_trips = state_trips[state_trips["TDWKND"] == 2]
            weekend_trips = state_trips[state_trips["TDWKND"] == 1]

            # Unique groups
            unique_weekday_ids = weekday_trips[["HOUSEID", "VEHID", "PERSONID", weight_column]].drop_duplicates()
            unique_weekend_ids = weekend_trips[["HOUSEID", "VEHID", "PERSONID", weight_column]].drop_duplicates()

            if unique_weekday_ids.empty or unique_weekend_ids.empty:
                print(f"Warning: No sufficient weekday or weekend data for {state_name}. Skipping...")
                continue

            # Normalize weights
            weekday_weights = normalize_weights(unique_weekday_ids, weight_column)
            weekend_weights = normalize_weights(unique_weekend_ids, weight_column)

            # Sample full groups
            sampled_weekday = sample_itinerary_groups(unique_weekday_ids, weekday_trips, required_n, weekday_weights)
            sampled_weekend = sample_itinerary_groups(unique_weekend_ids, weekend_trips, required_n, weekend_weights)

            # Assign NHOUSEID to each sampled group (same BOOT_ID -> same synthetic household)
            all_sampled = pd.concat([sampled_weekday, sampled_weekend], ignore_index=True)
            unique_sampled_ids = all_sampled[["HOUSEID", "VEHID", "PERSONID", "BOOT_ID"]].drop_duplicates()
            unique_sampled_ids["NHOUSEID"] = [f"{state_name}_{nhouseid_counter + i}" for i in range(len(unique_sampled_ids))]

            # Merge NHOUSEID back
            sampled_weekday = sampled_weekday.merge(unique_sampled_ids, on=["HOUSEID", "VEHID", "PERSONID", "BOOT_ID"], how="left")
            sampled_weekend = sampled_weekend.merge(unique_sampled_ids, on=["HOUSEID", "VEHID", "PERSONID", "BOOT_ID"], how="left")

            nhouseid_counter += len(unique_sampled_ids)

            # ✅ Do NOT touch the Day or Month columns
            final_itineraries.append(sampled_weekday)
            final_itineraries.append(sampled_weekend)

        # Combine and sort final output
        if final_itineraries:
            final_bootstrapped_data = pd.concat(final_itineraries, ignore_index=True)
            final_bootstrapped_data = final_bootstrapped_data.sort_values(by=["NHOUSEID", "PERSONID", "VEHID", "TDTRPNUM", "Day"])
        else:
            final_bootstrapped_data = pd.DataFrame()

        return final_bootstrapped_data

    # Apply bootstrapping with state name correction
    synthetic_itineraries = bootstrap_and_create_7_day_itineraries_fixed(new_itineraries, vehicle_data, state_abbrev_to_name, weight_column="WTTRDFIN")

    def create_final_7_day_itineraries(synthetic_itineraries, new_itineraries, vehicle_data, state_abbrev_to_name):
        # Step 1: Get original itineraries for states that did NOT require bootstrapping
        states_with_no_bootstrap = vehicle_data[vehicle_data["Additional Samples Needed"] == 0]["State"].tolist()
        state_name_to_abbrev = {v: k for k, v in state_abbrev_to_name.items()}
        state_abbrevs = [state_name_to_abbrev.get(state) for state in states_with_no_bootstrap if state_name_to_abbrev.get(state)]

        original_itineraries = new_itineraries[new_itineraries["HHSTATE"].isin(state_abbrevs)].copy()
        original_itineraries["State_Full"] = original_itineraries["HHSTATE"].map(state_abbrev_to_name)

        # Assign NHOUSEID to original itineraries
        original_groups = original_itineraries[["HOUSEID", "VEHID", "PERSONID", "State_Full"]].drop_duplicates()
        original_groups["NHOUSEID"] = [
            f"{row['State_Full']}_ORIG_{i}" for i, row in original_groups.iterrows()
        ]
        original_itineraries = original_itineraries.merge(
            original_groups, on=["HOUSEID", "VEHID", "PERSONID", "State_Full"], how="left"
        )

        # Step 2: Concatenate synthetic and original itineraries
        all_itineraries = pd.concat([synthetic_itineraries, original_itineraries], ignore_index=True)
        all_itineraries["NHOUSEID_n"] = all_itineraries["NHOUSEID"].astype("category").cat.codes + 1
        all_itineraries["NHOUSEID_n"] = all_itineraries["NHOUSEID_n"].astype("int")

        # Step 3: Generate 7-day itineraries
        final_7day = []
        skipped = 0
        # Create a pool of weekend itineraries grouped by state
        weekend_pool = all_itineraries[all_itineraries["TDWKND"] == 1].copy()
        weekend_pool["State_Full"] = weekend_pool["HHSTATE"].map(state_abbrev_to_name)
        weekend_groups = weekend_pool.groupby(["State_Full", "Income"])

        for nhouseid, group in all_itineraries.groupby("NHOUSEID_n"):
            weekday_trips = group[group["TDWKND"] == 2].copy()

            if weekday_trips.empty:
                skipped += 1
                continue

            state_name = group["HHSTATE"].map(state_abbrev_to_name).iloc[0]
            income_level = group["Income"].iloc[0]

            if (state_name, income_level) not in weekend_groups.groups:
                skipped += 1
                continue

            state_income_weekend_pool = weekend_groups.get_group((state_name, income_level))
            unique_weekends = state_income_weekend_pool[["VEHID", "PERSONID", "NHOUSEID_n"]].drop_duplicates()

            # Randomly select a weekend itinerary within the same state and income group
            weekend_id = unique_weekends.sample(n=1).iloc[0]
            weekend_group = state_income_weekend_pool[
                (state_income_weekend_pool["NHOUSEID_n"] == weekend_id["NHOUSEID_n"]) &
                (state_income_weekend_pool["VEHID"] == weekend_id["VEHID"]) &
                (state_income_weekend_pool["PERSONID"] == weekend_id["PERSONID"])
                ].copy()

            if weekend_group.empty:
                skipped += 1
                continue

            # Assign unified NHOUSEID to weekend as well
            weekend_group["NHOUSEID_n"] = nhouseid

            # Helper: shift day values sequentially
            def shift_days(df, repeat_count, start_day):
                original_days = df["Day"].values
                full = []
                for i in range(repeat_count):
                    shifted = df.copy()
                    day_offset = i * (original_days.max() - original_days.min() + 1)
                    shifted["Day"] = original_days + start_day + day_offset - original_days.min()
                    full.append(shifted)
                return pd.concat(full, ignore_index=True)

            # Tile weekdays and weekends
            weekday_shifted = shift_days(weekday_trips, 5, start_day=1)
            max_weekday_day = weekday_shifted["Day"].max()
            weekend_shifted = shift_days(weekend_group, 2, start_day=max_weekday_day + 1)

            # Fix Month overflow
            start_month = group["Month"].dropna().astype(int).iloc[0] if not group["Month"].dropna().empty else 1
            days_in_month = calendar.monthrange(2017, start_month)[1]
            weekday_shifted["Month"] = start_month
            weekend_shifted["Month"] = start_month

            if weekend_shifted["Day"].max() > days_in_month:
                weekend_shifted["Month"] += 1
                weekend_shifted["Day"] -= days_in_month

            final_7day.append(pd.concat([weekday_shifted, weekend_shifted], ignore_index=True))

        final_df = pd.concat(final_7day, ignore_index=True)
        final_df = final_df.sort_values(by=["NHOUSEID_n", "PERSONID", "VEHID", "Day", "TDTRPNUM"])

        return final_df

    combined_itinararies = create_final_7_day_itineraries(synthetic_itineraries, new_itineraries, vehicle_data, state_abbrev_to_name)
    combined_itinararies["NHOUSEID_n"] = combined_itinararies["NHOUSEID_n"].astype("int")

    def save_itineraries_grouped_by_state(combined_itineraries, output_dir):
        states = combined_itineraries["HHSTATE"].unique()

        for state in tqdm(states, desc="Processing states"):
            state_df = combined_itineraries[combined_itineraries["HHSTATE"] == state].copy()
            grouped = state_df.groupby(["NHOUSEID_n"])

            itineraries_day = []
            for _, group in tqdm(grouped, desc=f"Packaging itineraries for {state}", leave=False):
                itinerary_dict_day = {'trips': group}
                itineraries_day.append(itinerary_dict_day)

            output_file_day = f"{output_dir}/itineraries_{state}.pkl"
            with open(output_file_day, 'wb') as f:
                pkl.dump(itineraries_day, f)

            print(f"Saved {len(itineraries_day)} itineraries for {state} to {output_file_day}")
            del itineraries_day  # free memory after each state

    save_itineraries_grouped_by_state(combined_itinararies, output_direction)

    # Display time taken
    print('Itineraries creation took {:.4f} seconds'.format(time.time() - t0))


def MakeItineraries_eVMT():
    codex_path = 'D:\\Hanif\\V2G_national\\charging\\codex.json'
    output_file = 'D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data\\itineraries_evmt.pkl'
    # Load input files and initial data
    data = process.load(codex_path)
    data_charging = data.get("charging", pd.DataFrame())
    data_trips = data.get("trips", pd.DataFrame())
    data_trips = data_trips.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    chevrolet_vehids = data_trips[data_trips['Make'].str.lower() == 'chevrolet']['vehicle_name'].unique().tolist()

    temp = final_temp_adjustment()

    # Define a function to extract battery capacity
    def get_battery_capacity(row):
        if 'Leaf' in row['vehicle_model']:
            return int(row['vehicle_model'].split()[1])
        elif 'Bolt' in row['vehicle_model']:
            return 66
        elif 'RAV4 EV' in row['vehicle_model']:
            return 41.8
        elif 'Model S' in row['Model']:
            # Extract capacity from the Model column if available
            model_parts = row['Model'].split()
            for part in model_parts:
                try:
                    return float(part.strip('P').strip('D'))
                except ValueError:
                    continue
            # If not found, use the vehicle_model and take the lower number
            if 'Model S' in row['Model']:
                # Replace underscores with spaces and extract numeric parts
                clean_model = row['vehicle_model'].replace('_', ' ')
                capacities = [float(s) for s in clean_model.split() if s.replace('.', '').isdigit()]
                if capacities:
                    return max(capacities)
        elif 'Model s' in row['Model']:
            # Extract capacity from the Model column if available
            model_parts = row['Model'].split()
            for part in model_parts:
                try:
                    return float(part.strip('P').strip('D'))
                except ValueError:
                    continue
            # If not found, use the vehicle_model and take the lower number
            if 'Model s' in row['Model']:
                # Replace underscores with spaces and extract numeric parts
                clean_model = row['vehicle_model'].replace('_', ' ')
                capacities = [float(s) for s in clean_model.split() if s.replace('.', '').isdigit()]
                if capacities:
                    return max(capacities)
        # Return NaN if no match
        return np.nan

    data_trips['battery_capacity_kwh'] = data_trips.apply(get_battery_capacity, axis=1)
    # Assign energy consumption values for California
    data_trips["Energy_Consumption"] = temp.loc[temp["HHSTATE"] == "CA", "Energy_Consumption"].iloc[0]
    data_trips["Energy_Consumption"] = data_trips["Energy_Consumption"] * 2236.94185

    # Function to convert datetime columns to HHMM format
    def convert_start_end_to_HHMM(df, start_column, end_column):
        def extract_HHMM(timestamp):
            try:
                ts = pd.Timestamp(timestamp)
                return int(f"{ts.hour}{ts.minute:02d}")
            except Exception:
                return None  # Handle invalid timestamps gracefully

        df["start_HHMM"] = df[start_column].apply(extract_HHMM)
        df["end_HHMM"] = df[end_column].apply(extract_HHMM)

        return df[["start_HHMM", "end_HHMM"]]

    # Rename columns for clarity and consistency
    data_trips.rename(columns={"Household": "HOUSEID", "duration": "TRVLCMIN", "distance": "TRPMILES", "vehicle_name": "VEHID", "battery_capacity_kwh": "BATTCAP"}, inplace=True)
    data_trips[["STRTTIME", "ENDTIME"]] = convert_start_end_to_HHMM(data_trips, "start_time_ (local)", "end_time_ (local)")
    data_trips.loc[data_trips["destination_label"] == "Home", "WHYTRP1S"] = 1
    data_trips.loc[data_trips["destination_label"] == "Work", "WHYTRP1S"] = 10
    data_trips.loc[((data_trips["destination_label"] != "Work") & (data_trips["destination_label"] != "Home")), "WHYTRP1S"] = 20
    data_trips["next_start_time"] = data_trips.groupby("VEHID")["start_time_ (local)"].shift(-1)
    data_trips["start_time_ (local)"] = pd.to_datetime(data_trips["start_time_ (local)"], errors='coerce')
    data_trips["next_start_time"] = pd.to_datetime(data_trips["next_start_time"], errors='coerce')
    data_trips["DWELTIME"] = ((data_trips["next_start_time"] - data_trips["start_time_ (local)"]).dt.total_seconds()) / 60
    data_trips.loc[data_trips["DWELTIME"] < 0, "DWELTIME"] = 6000
    data_trips.loc[data_trips["DWELTIME"] == 0, "DWELTIME"] = 1
    data_trips["TRVLCMIN"] = data_trips["TRVLCMIN"] / 60
    data_trips.loc[data_trips["TRVLCMIN"] < 1, "TRVLCMIN"] = 1
    data_trips = data_trips[data_trips["TRVLCMIN"] < 480]
    data_trips.loc[data_trips["DWELTIME"].isna(), "DWELTIME"] = 6000
    data_trips.loc[data_trips["day_type"] == "weekday", "TDWKND"] = 2
    data_trips.loc[data_trips["day_type"] == "weekend", "TDWKND"] = 1
    data_trips["HHSTATE"] = "CA"
    data_trips["HHSTFIPS"] = 6
    data_trips["Income"] = 175000
    data_trips.loc[data_trips["destination_label"] == "Home", "WHYTRP1S"] = 1
    data_trips.loc[data_trips["destination_label"] == "Work", "WHYTRP1S"] = 10
    data_trips.loc[data_trips["destination_label"] == "Other", "WHYTRP1S"] = 100
    data_trips = data_trips.loc[(data_trips["Model"] != "Leaf") & (data_trips["Model"] != "RAV4 EV")]
    # Prepare itineraries
    new_itineraries = data_trips
    new_itineraries = new_itineraries[["HOUSEID", "year", "month", "day", "STRTTIME", "ENDTIME", "VEHID", "TRVLCMIN", "TRPMILES", "DWELTIME", "WHYTRP1S", "TDWKND", "HHSTATE", "Income", "Energy_Consumption", "BATTCAP"]]
    new_itineraries.rename(columns={"year": "Year", "month": "Month", "day": "Day"}, inplace=True)
    new_itineraries.loc[(new_itineraries["TRPMILES"] > 120) & (new_itineraries['VEHID'].isin(chevrolet_vehids)), "TRPMILES"] = 120
    new_itineraries.loc[(new_itineraries["TRPMILES"] > 120) & (~new_itineraries['VEHID'].isin(chevrolet_vehids)), "TRPMILES"] = 120
    unique_combinations = new_itineraries[["HOUSEID", "VEHID"]].drop_duplicates()

    itineraries = np.array([None] * unique_combinations.shape[0])
    t0 = time.time()

    # Main loop: iterate over each unique household-vehicle-person combination
    for idx, row in tqdm(enumerate(unique_combinations.itertuples(index=False)), total=unique_combinations.shape[0]):
        hh_id = row.HOUSEID
        veh_id = row.VEHID
        # Get trips for this specific household-vehicle-person combination
        trips_indices = np.argwhere((new_itineraries['HOUSEID'] == hh_id) & (new_itineraries['VEHID'] == veh_id)).flatten()
        # Create the dictionary for the current household-vehicle-person combination
        itinerary_dict = {'trips': new_itineraries.iloc[trips_indices]}
        # Store in the array at the current index
        itineraries[idx] = itinerary_dict

    # Display time taken
    print('Itineraries creation took {:.4f} seconds'.format(time.time() - t0))

    # Save itineraries to file
    t0 = time.time()
    print('Pickling outputs:', end='')
    with open(output_file, 'wb') as f:
        pkl.dump(itineraries, f)
    print(' Done in {:.4f} seconds'.format(time.time() - t0))

# %%
# MakeItineraries_bootstrap(confidence=0.95, output_direction="D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data")
# MakeItineraries(day_type="all")
# MakeItineraries(day_type="weekend")
# MakeItineraries_eVMT()


# %%
#
with open("D:\\Hanif\\V2G_national\\charging\\Data\\Generated_Data\\itineraries_TX.pkl", "rb") as f:
    data = pickle.load(f)
test = data[0]["trips"]
all_trips = pd.concat([entry["trips"] for entry in data], ignore_index=True)


