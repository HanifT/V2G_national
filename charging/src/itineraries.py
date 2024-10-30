import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from tqdm import tqdm
# %%


def MakeItineraries():

	# loading in data
	t0 = time.time()
	print('Loading NHTS data:', end='')
	trips = pd.read_csv('/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub.csv')
	dmv = pd.read_csv('/Users/haniftayarani/V2G_national/charging/Data/dmv.csv')
	trips = trips.drop([
		"TRPACCMP", "TRPHHACC", 'TRWAITTM', 'NUMTRANS', 'TRACCTM', 'DROP_PRK', 'TREGRTM', 'WHODROVE',
		'HHMEMDRV', 'HH_ONTD', 'NONHHCNT', 'NUMONTRP', 'PSGR_FLG', 'DRVR_FLG', 'ONTD_P1', 'ONTD_P2', 'ONTD_P3',
		'ONTD_P4', 'ONTD_P5', 'ONTD_P6', 'ONTD_P7', 'ONTD_P8', 'ONTD_P9', 'ONTD_P10', 'ONTD_P11', 'ONTD_P12',
		'ONTD_P13', 'TRACC_WLK', 'TRACC_POV', 'TRACC_BUS', 'TRACC_CRL', 'TRACC_SUB', 'TRACC_OTH', 'TREGR_WLK',
		'TREGR_POV', 'TREGR_BUS', 'TREGR_CRL', 'TREGR_SUB', 'TREGR_OTH', 'DRVRCNT', 'NUMADLT', 'WRKCOUNT',
		'HHRESP', 'LIF_CYC', 'MSACAT', 'MSASIZE', 'RAIL', 'HH_RACE', 'HH_HISP', 'HH_CBSA', 'SMPLSRCE', 'R_AGE',
		'EDUC', 'R_SEX', 'PRMACT', 'PROXY', 'WORKER', 'DRIVER', 'WTTRDFIN', 'WHYTRP90', 'TRPMILAD', 'R_AGE_IMP',
		'R_SEX_IMP', 'OBHUR', 'DBHUR', 'OTHTNRNT', 'OTPPOPDN', 'OTRESDN', 'OTEEMPDN', 'OBHTNRNT', 'OBPPOPDN',
		'OBRESDN', 'DTHTNRNT', 'DTPPOPDN', 'DTRESDN', 'DTEEMPDN', 'DBHTNRNT', 'DBPPOPDN', 'DBRESDN'
	], axis=1)
	trip_veh = [1, 2, 3, 4, 5, 6]
	veh_type = [-1, 1, 2, 3, 4, 5]
	trips = trips[trips["TRPTRANS"].isin(trip_veh)]
	trips = trips[trips["TRPMILES"] > 0]
	trips = trips[trips["TRVLCMIN"] > 0]
	trips = trips[trips["VEHTYPE"].isin(veh_type)]
	trips = trips[trips["TRPMILES"] < 500]

	# Step 1: Separate weekday trips (TRAVDAY between 2 and 6) and weekend trips (TRAVDAY is 1 or 7)
	weekday_trips = trips[(trips['TRAVDAY'] >= 2) & (trips['TRAVDAY'] <= 6)].copy()
	weekend_trips = trips[(trips['TRAVDAY'] == 1) | (trips['TRAVDAY'] == 7)].copy()

	# Step 2: Replicate each weekday trip for each weekday (Monday to Friday)
	expanded_trips = pd.concat([weekday_trips] * 5, ignore_index=True)

	# Step 3: Update the TRAVDAY values for expanded trips to cycle through Monday to Friday (2 to 6)
	expanded_trips['TRAVDAY'] = (expanded_trips.index % 5) + 2

	# Step 4: Combine the expanded weekday trips with the unchanged weekend trips
	new_itineraries = pd.concat([expanded_trips, weekend_trips], ignore_index=True)

	# Step 5: Sort by HOUSEID, PERSONID, VEHID, and TRAVDAY to organize the final DataFrame
	new_itineraries = new_itineraries.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', 'TRAVDAY', "TDTRPNUM"]).reset_index(drop=True)
	# Group by HHSTATE and count unique combinations of HOUSEID, PERSONID, and VEHID for each state

	print(' {:.4f} seconds'.format(time.time()-t0))

	t0 = time.time()
	print('Creating itinerary dicts:', end='')

	# Selecting for household vehicles
	new_itineraries = new_itineraries[(
		(new_itineraries['TRPHHVEH'] == 1) &
		(new_itineraries['TRPTRANS'] >= 3) &
		(new_itineraries['TRPTRANS'] <= 6)
		)].copy()

	# Get unique combinations of household, vehicle, and person IDs
	unique_combinations = new_itineraries[['HOUSEID', 'VEHID', 'PERSONID']].drop_duplicates()

	# Start timer
	t0 = time.time()

	# Initialize an array to store each itinerary dictionary
	itineraries = np.array([None] * unique_combinations.shape[0])

	# Main loop: iterate over each unique household-vehicle-person combination in the test set
	for idx, row in tqdm(enumerate(unique_combinations.itertuples(index=False))):
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

	# Save itineraries to a pickle file
	output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries.pkl'
	t0 = time.time()
	print('Pickling outputs:', end='')
	pkl.dump(itineraries, open(output_file, 'wb'))
	print(' Done in {:.4f} seconds'.format(time.time() - t0))

# %%
# MakeItineraries()
#
# # %%
# # Specify the path to the pickle file
# input_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries.pkl'
#
# # Load the pickle file
# t0 = time.time()
# print('Loading pickled outputs:', end='')
# with open(input_file, 'rb') as file:
#     itineraries = pkl.load(file)
# print(' Done in {:.4f} seconds'.format(time.time() - t0))
#
# # Extract 'trips' DataFrames and concatenate into a single DataFrame
# trips_df = pd.concat([itinerary['trips'] for itinerary in itineraries if itinerary is not None], ignore_index=True)
#
#
# # Test output: print the first entry in the loaded data
# print(itineraries[3])  # Adjust index if needed, depending on your data structure
