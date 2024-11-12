import os
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _print(string, disp=True):

    if disp:

        print(string)

def load(codex_path, verbose = False):
	'''
	Load data based on a codex file. the codex file will be a JSON of the below format:

	[
		{
			"name": <variable name>,
			"file": <file name>,
			"kwargs": {
				<argument>: <value>,
				<argument>: <value>
			}
		}
	]

	Files must be CSV or Excel types. Files will be loaded using the appropriate Pandas
	method with **kwargs as inputs.

	Output will be a dictionary of {<name>: <DataFrame>}
	'''

	t0 = time.time()

	# Getting file parts for codex

	codex_directory, codex_file = os.path.split(codex_path)

	# Loading codex file

	with open(codex_path, 'r') as file:

		codex = json.load(file)

	# Loading in data

	data = {}

	for item in codex:

		extension = item['file'].split('.')[-1]

		load_path = os.path.join(codex_directory, item['file'])

		if extension == 'csv':

			data[item['name']] = pd.read_csv(load_path, **item['kwargs'])

		elif (extension == 'xlsx') or (extension == 'xls'):

			data[item['name']] = pd.read_excel(load_path, **item['kwargs'])

		else:

			raise RawDataFileTypeException

	_print(f'Data loaded: {time.time() - t0:.4} seconds', disp = verbose)

	return data

class RawDataFileTypeException(Exception):

	"Raw data input files must be .csv, .xls, or .xlsx"

	pass

def plot_charging_demand(df, plot_type='total'):
    """
    Plots the charging demand for households with different charger types (Level 1 Only, Level 2 Only, Both Levels)
    at home and non-home locations.

    Parameters:
    - df (dict): Dictionary containing data frames for Level 1 Only, Level 2 Only, and Both Levels charging demand.
      Each data frame should be a dictionary with 'home' and 'non_home' keys.
    - plot_type (str): Type of plot. Options are 'total' for total energy demand or 'average' for average demand per vehicle.
    """
    # Unpack data frames
    level1_home, level1_non_home = df['Level 1 Only']
    level2_home, level2_non_home = df['Level 2 Only']
    both_home, both_non_home = df['Both Levels']

    # Calculate average if requested
    if plot_type == 'average':
        level1_home['total_energy'] /= level1_home['vehicle_name'].nunique()
        level1_non_home['total_energy'] /= level1_non_home['vehicle_name'].nunique()
        level2_home['total_energy'] /= level2_home['vehicle_name'].nunique()
        level2_non_home['total_energy'] /= level2_non_home['vehicle_name'].nunique()
        both_home['total_energy'] /= both_home['vehicle_name'].nunique()
        both_non_home['total_energy'] /= both_non_home['vehicle_name'].nunique()

    # Update the 'Label' values to show only three main categories
    level1_home['Label'] = 'Level 1 Only'
    level1_non_home['Label'] = 'Level 1 Only'
    level2_home['Label'] = 'Level 2 Only'
    level2_non_home['Label'] = 'Level 2 Only'
    both_home['Label'] = 'Both Levels'
    both_non_home['Label'] = 'Both Levels'

    # Concatenate all DataFrames with updated labels
    combined_df = pd.concat([
        level1_home.assign(Location='Home'),
        level1_non_home.assign(Location='Non-Home'),
        level2_home.assign(Location='Home'),
        level2_non_home.assign(Location='Non-Home'),
        both_home.assign(Location='Home'),
        both_non_home.assign(Location='Non-Home')
    ])

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Label', y='total_energy', hue='Location', data=combined_df)

    # Set plot labels
    plt.xlabel('Home Charger Type')
    plt.ylabel('Total Energy Demand (kWh) per Vehicle' if plot_type == 'total' else 'Average Energy Demand per Vehicle (kWh)')
    plt.title('Home vs Non-Home Charging Demand by Household Charger Type' if plot_type == 'total' else
              'Home vs Non-Home Average Charging Demand per Vehicle by Household Type')
    plt.xticks(rotation=45)
    plt.legend(title='Location')
    plt.tight_layout()
    plt.show()


def plot_days_between_charges(data, min_energy=2, min_days=0, max_days=10, excluded_makes=["Nissan", "Toyota"]):
    """
    Filters data, calculates days between charging sessions, and plots the distribution of days between charges by vehicle model.

    Parameters:
    - data (DataFrame): The DataFrame containing charging data with 'Make', 'vehicle_name', 'start_time_ (local)', 'total_energy', and 'Model' columns.
    - min_energy (float): Minimum energy threshold for filtering charging sessions.
    - min_days (int): Minimum days between charges to consider in the plot.
    - max_days (int): Maximum days between charges to consider in the plot.
    - excluded_makes (list): List of vehicle makes to exclude from the analysis.

    Returns:
    - None: Displays a box plot of days between charges for each vehicle model.
    """
    # Exclude specific vehicle makes
    data_filtered = data[~data["Make"].isin(excluded_makes)]

    # Convert 'start_time_ (local)' column to datetime format
    data_filtered['start_time_ (local)'] = pd.to_datetime(data_filtered['start_time_ (local)'], format='%m/%d/%y %H:%M')

    # Create a new column for the day of the year
    data_filtered['day_of_year'] = data_filtered['start_time_ (local)'].dt.dayofyear

    # Group by vehicle_name and day_of_year, then get the first charging session time for each day
    df_grouped = data_filtered.groupby(['vehicle_name', 'day_of_year']).first().reset_index()

    # Filter out sessions with less than the specified minimum energy
    df_grouped = df_grouped[df_grouped["total_energy"] > min_energy]

    # Sort by vehicle and time to ensure chronological order
    df_grouped = df_grouped.sort_values(by=['vehicle_name', 'start_time_ (local)'])

    # Calculate the number of days between each charging session for each vehicle
    df_grouped['days_between_charges'] = df_grouped.groupby('vehicle_name')['day_of_year'].diff()

    # Fill NaN values in 'days_between_charges' with 0, indicating the first session
    df_grouped['days_between_charges'] = df_grouped['days_between_charges'].fillna(0)

    # Filter based on days between charges
    df_grouped = df_grouped[(df_grouped['days_between_charges'] > min_days) & (df_grouped['days_between_charges'] < max_days)]

    # Plot box plot of days between charges grouped by vehicle model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_grouped, x='Model', y='days_between_charges')

    # Customize plot labels and title
    plt.xticks(rotation=45)
    plt.xlabel("Vehicle Model")
    plt.ylabel("Days Between Charges")
    plt.title("Distribution of Days Between Charging Sessions by Vehicle Model")

    # Use tight layout to fit all elements within the figure
    plt.tight_layout()

    # Display the plot
    plt.show()

    return data_filtered, df_grouped



def add_order_column(df):
    # Use groupby and cumcount to create the 'order' column
    df['order'] = df.groupby(['HOUSEID', 'PERSONID', 'VEHID']).cumcount() + 1

    return df


def assign_days_of_week(df):
    # Define the days of the week
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Sort by HOUSEID, PERSONID, VEHID, and STRTTIME to ensure chronological order
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', "order"]).reset_index(drop=True)

    # Initialize an empty list to store the day of the week for each trip
    day_of_week_column = []

    # Initialize variables to track the previous STRTTIME and current day index for each unique vehicle
    prev_strttime = None
    day_index = 0

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Check if this row is the first entry for a new HOUSEID-PERSONID-VEHID combination
        if i == 0 or (
                row['HOUSEID'] != df.at[i - 1, 'HOUSEID'] or
                row['PERSONID'] != df.at[i - 1, 'PERSONID'] or
                row['VEHID'] != df.at[i - 1, 'VEHID']
        ):
            # Start a new week for this unique vehicle
            day_index = 0
            day_of_week_column.append(days_of_week[day_index])
        else:
            # Check if STRTTIME is less than the previous time, indicating a new day
            if row['STRTTIME'] < prev_strttime:
                day_index = (day_index + 1) % 7  # Move to the next day, wrap around after Sunday

            # Append the current day of the week to the list
            day_of_week_column.append(days_of_week[day_index])

        # Update the previous STRTTIME to the current one
        prev_strttime = row['STRTTIME']

    # Add the day_of_week_column as a new column in the DataFrame
    df['days_of_week'] = day_of_week_column

    return df


def identify_charging_sessions(df):

    # Sort by HOUSEID, PERSONID, VEHID, and order to ensure chronological order within each group
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', 'order']).reset_index(drop=True)

    # Calculate the difference in SOC within each unique vehicle group
    df['charging'] = df.groupby(['HOUSEID', 'PERSONID', 'VEHID'])['SOC'].diff().fillna(0) > 0

    return df

def calculate_days_between_charges_synt(df):
    # Define a mapping from days of the week to numeric values
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}

    # Map 'days_of_week' to numeric values for easy calculation
    df['day_numeric'] = df['days_of_week'].map(day_mapping)

    # Sort by HOUSEID, PERSONID, VEHID, and 'order' to ensure chronological order within each group
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', 'order']).reset_index(drop=True)

    # Initialize a list to store the days between charges for each row
    days_between_charges = []

    # Track the last charging day for each unique vehicle
    last_charging_day = {}

    # Iterate through each row
    for i, row in df.iterrows():
        # Create a unique key for each vehicle
        vehicle_key = (row['HOUSEID'], row['PERSONID'], row['VEHID'])

        # Check if this row is the first entry for the vehicle
        if vehicle_key not in last_charging_day:
            # Set days_between_charges to 0 for the first entry
            days_diff = 0
            last_charging_day[vehicle_key] = row['day_numeric']
        else:
            # For subsequent entries, check if this is a charging event
            if row['charging'] == True:
                # Calculate the difference in days, adjusting for week wrap-around
                days_diff = (row['day_numeric'] - last_charging_day[vehicle_key]) % 7
                # Update the last charging day for this vehicle
                last_charging_day[vehicle_key] = row['day_numeric']
            else:
                # If it's not a charging event, set days_diff to None
                days_diff = None

        # Append the calculated days_diff to the list
        days_between_charges.append(days_diff)

    # Add the 'days_between_charges' column to the DataFrame and drop the 'day_numeric' helper column
    df['days_between_charges'] = days_between_charges
    df = df.drop(columns=['day_numeric'])

    # Filter out None values and the first row (order == 1) for each unique vehicle for plotting
    df_filtered = df[(df['days_between_charges'].notna()) & (df['order'] != 1)]
    # Plot the box plot for days_between_charges
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x='days_between_charges')
    plt.xlabel("Days Between Charges")
    plt.title("Distribution of Days Between Charges")
    plt.show()

    return df, df_filtered
