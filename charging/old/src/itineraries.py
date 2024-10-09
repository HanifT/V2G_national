import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from tqdm import tqdm

def MakeItineraries(
	vehicles_file='../Data/NHTS_2017/vehpub.csv',
	trips_file='../Data/NHTS_2017/trippub.csv',
	out_file='../Data/Generated_Data/itineraries.pkl',
	n_households=0,
	disp_freq=10
	):

	#loading in data
	t0=time.time()
	print('Loading NHTS data:',end='')
	vehicles=pd.read_csv(vehicles_file)
	trips=pd.read_csv(trips_file)
	print(' {:.4f} seconds'.format(time.time()-t0))

	t0=time.time()
	print('Creating itinerary dicts:',end='')
	#Selecting for household vehicles
	trips=trips[(
		(trips['TRPHHVEH']==1)&
		(trips['TRPTRANS']>=3)&
		(trips['TRPTRANS']<=6)&
		(trips['VEHID']<=2)
		)].copy()

	#all unique vehicles
	hh_ids=trips['HOUSEID'].copy().to_numpy()
	veh_ids=trips['VEHID'].copy().to_numpy()

	#optionally down-selecting data to n random households
	

	if n_households:
		unique_hh,unique_hh_inverse=np.unique(hh_ids,return_inverse=True)
		
		hh_keep=np.random.randint(0,len(unique_hh)-1,n_households)
		idx_keep=np.isin(hh_ids,unique_hh[hh_keep])
		hh_ids=hh_ids[idx_keep]
		veh_ids=veh_ids[idx_keep]

	# unique_hh,unique_hh_counts=np.unique(hh_ids,return_counts=True)

	# idx_keep=np.isin(hh_ids,unique_hh[unique_hh_counts>=2])
	# hh_ids=hh_ids[idx_keep]
	# veh_ids=veh_ids[idx_keep]

	itineraries=np.array([None]*hh_ids.shape[0])
	# keep=np.array([False]*hh_ids.shape[0])

	#Main loop
	for idx in tqdm(range(hh_ids.shape[0])):
		hh_id=hh_ids[idx]
		veh_id=veh_ids[idx]
		trips_indices=np.argwhere(
			(trips['HOUSEID']==hh_id)&(trips['VEHID']==veh_id)).flatten()
		# print(trips_indices,hh_id,veh_id)
		# print('a',sum(trips_indices))
		# break
		# keep[idx]=sum(trips_indices)>=2
		itineraries[idx]=({
			'trips':trips.iloc[trips_indices]
			})

	print(' {:.4f} seconds'.format(time.time()-t0))

	t0=time.time()
	print('Pickling outputs:',end='')
	pkl.dump(itineraries,open(out_file,'wb'))
	print(' {:.4f} seconds'.format(time.time()-t0))

class Itinerary():

	def __init__(self,hh_id,veh_id,veh,trips):

		self.veh=veh.to_dict(orient='records')[0]
		self.trips=trips

#Function for importing, cleaning, and processing trips data
def LoadNHTSData(trips_file):
	#Loading in the trips data
	trips_df=pd.read_csv(trips_file)
	#Filtering out non-vehicle trips
	trips_df=trips_df[(trips_df['VEHID']>0)&(trips_df['VEHID']<12)]
	#Removing unnecessary columns
	trips_df=trips_df[(['HOUSEID','VEHID','STRTTIME','ENDTIME',
		'TRVLCMIN','TRPMILES','TRPTRANS','DWELTIME','WHYTRP1S',
		'OBHUR','HHSTFIPS','HH_CBSA'])]
	return trips_df



def MakeNHTSItineraries(in_file='../Data/NHTS_2017/trippub.csv',
	out_file='../Data/Generated_Data/NHTS_Itineraries.pkl'):
	
	t0=time.time()
	print('Loading NHTS data:',end='')
	NHTS_df=LoadNHTSData(in_file)
	print(' {:.4f} seconds'.format(time.time()-t0))

	t0=time.time()
	print('Creating itinerary objects:',end='')
	NHTS_itineraries=CreateItineraries(NHTS_df)
	print(' {:.4f} seconds'.format(time.time()-t0))

	t0=time.time()
	print('Pickling outputs:',end='')
	pkl.dump(NHTS_itineraries,open(out_file,'wb'))
	print(' {:.4f} seconds'.format(time.time()-t0))

	print('Done')

if __name__ == "__main__":
	argv=sys.argv
	if len(argv)>=2:
		MakeItineraries(n_households=int(argv[1]))
	else:
		MakeItineraries()
