import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import time
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from temp import final_temp_adjustment
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
# %%

def MakeItineraries(day_type="weekday"):

	# Start timer
	t0 = time.time()
	print('Loading NHTS data:', end='')

	# Load data
	trips = pd.read_csv('/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub.csv')
	temp = final_temp_adjustment()
	trips = pd.merge(trips, temp[["Energy_Consumption", "HHSTATE", "HHSTFIPS"]], on=["HHSTATE", "HHSTFIPS"], how="left")
	# Drop unnecessary columns
	trips = trips[['HOUSEID', 'PERSONID', 'TDTRPNUM', 'STRTTIME', 'ENDTIME', 'TRVLCMIN',
       'TRPMILES', 'TRPTRANS', 'VEHID', 'WHYFROM', 'LOOP_TRIP', 'TRPHHVEH',
       'PUBTRANS', 'TRIPPURP', 'DWELTIME', 'TDWKND', 'VMT_MILE', 'WHYTRP1S',
       'TDCASEID', 'WHYTO', 'TRAVDAY', 'HOMEOWN', 'HHSIZE', 'HHVEHCNT',
       'HHFAMINC', 'HHSTATE', 'HHSTFIPS', 'TDAYDATE', 'URBAN', 'URBANSIZE',
       'URBRUR', 'GASPRICE', 'CENSUS_D', 'CENSUS_R', 'CDIVMSAR', 'VEHTYPE',"Energy_Consumption"]]

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
	trips = trips[trips["TRPMILES"] < 500]
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

	# Initialize an array to store each itinerary dictionary
	itineraries = np.array([None] * unique_combinations.shape[0])

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
		output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
	elif day_type == "weekend":
		output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'
	elif day_type == "all":
		output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl'

	# Save itineraries to a pickle file
	t0 = time.time()
	print('Pickling outputs:', end='')
	pkl.dump(itineraries, open(output_file, 'wb'))
	print(' Done in {:.4f} seconds'.format(time.time() - t0))

# %%
# MakeItineraries(day_type="all")
# MakeItineraries(day_type="weekday")
# MakeItineraries(day_type="weekend")


