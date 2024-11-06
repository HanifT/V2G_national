import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
# %%


def MakeItineraries(weekday=True):

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

	# Apply the TDWKND filter based on the weekday parameter
	if weekday:
		trips = trips[trips["TDWKND"] == 2]  # Select weekdays
	else:
		trips = trips[trips["TDWKND"] == 1]  # Select weekends

	trip_veh = [1, 2, 3, 4, 5, 6]
	veh_type = [-1, 1, 2, 3, 4, 5]

	trips = trips[trips["TRPTRANS"].isin(trip_veh)]
	trips = trips[trips["TRPMILES"] > 0]
	trips = trips[trips["TRVLCMIN"] > 0]
	trips = trips[trips["VEHTYPE"].isin(veh_type)]
	trips = trips[trips["TRPMILES"] < 500]

	new_itineraries = trips
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
	# Set output file name based on weekday parameter
	if weekday:
		output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
	else:
		output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'

	t0 = time.time()
	print('Pickling outputs:', end='')
	pkl.dump(itineraries, open(output_file, 'wb'))
	print(' Done in {:.4f} seconds'.format(time.time() - t0))
# %%
MakeItineraries(weekday=True)
MakeItineraries(weekday=False)

