import sys
import time
import signal
import numpy as np
from scipy.stats import t
import os
import pandas as pd
import pickle
from contextlib import contextmanager
from scipy.special import comb


'''
Execution time limiter from:
https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
'''
class TimeoutException(Exception): pass

@contextmanager
def TimeLimit(seconds):
	def SignalHandler(signum,frame):
		raise TimeoutException()
	signal.signal(signal.SIGALRM, SignalHandler)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)

continental_us_fips=([1,4,5,6,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,
	27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,
	51,53,54,55,56])

us_state_fips=([1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,
	27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,
	51,53,54,55,56])

alaska_fips=2
hawaii_fips=15

continental_us_abb=(['AL','AZ','AR','CA','CO','CT','DE','DC','FL','GA','ID','IL',
	'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
	'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
	'VT','VA','WA','WV','WI','WY'])

us_state_abb=(['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL',
	'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
	'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
	'VT','VA','WA','WV','WI','WY'])

'''
Calculates Gini coefficient (inequality)
'''
def GiniCoefficient(x):

	total=0

	for i,xi in enumerate(x[:-1],1):
		total+=np.sum(np.abs(xi-x[i:]))

	return total/(len(x)**2*np.mean(x))

def BinomialDistribution(n,r,p):
	return comb(n,r)*p**r*(1-p)**(n-r)

def IsIterable(value):
	return hasattr(value,'__iter__')

def TopNIndices(array,n):
	return sorted(range(len(array)), key=lambda i: array[i])[-n:]

def BottomNIndices(array,n):
	return sorted(range(len(array)), key=lambda i: array[i])[:n]

def T_Test(x,y,alpha):
	x_n=len(x)
	y_n=len(y)
	x_mu=x.mean()
	y_mu=y.mean()
	x_sig=x.std()
	y_sig=y.std()
	x_se=x_sig/np.sqrt(x_n)
	y_se=y_sig/np.sqrt(y_n)
	x_y_se=np.sqrt(x_se**2+y_se**2)
	T=(x_mu-y_mu)/x_y_se
	DF=x_n+y_n
	T0=t.ppf(1-alpha,DF)
	P=(1-t.cdf(np.abs(T),DF))*2
	return (P<=alpha),T,P,T0,DF

def FullFact(levels):
	n = len(levels)  # number of factors
	nb_lines = np.prod(levels)  # number of trial conditions
	H = np.zeros((nb_lines, n))
	level_repeat = 1
	range_repeat = np.prod(levels).astype(int)
	for i in range(n):
		range_repeat /= levels[i]
		range_repeat=range_repeat.astype(int)
		lvl = []
		for j in range(levels[i]):
			lvl += [j]*level_repeat
		rng = lvl*range_repeat
		level_repeat *= levels[i]
		H[:, i] = rng
	return H.astype(int)

def Pythagorean(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2+(y1-y2)**2)

#Function for calculating distances between lon/lat pairs
def Haversine(lon1,lat1,lon2,lat2):
	r=6372800 #[m]
	dLat=np.radians(lat2-lat1)
	dLon=np.radians(lon2-lon1)
	lat1=np.radians(lat1)
	lat2=np.radians(lat2)
	a=np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c=2*np.arcsin(np.sqrt(a))
	return c*r

def RMSE(x,y):

	return np.sqrt(((x-y)**2).sum()/len(x))

def CondPrint(message,disp=True):

	if disp:
		print(message)


#Custom progress bar
class ProgressBar():

	def __init__(self,iterable,bar_length=20,disp=True,freq=1):

		self.iterable=iterable
		self.total=len(iterable)
		self.bar_length=bar_length
		self.disp=disp
		self.freq=freq
		
		if self.disp:
			self.update=self.Update
		else:
			self.update=self.Update_Null

	def __iter__(self):

		return PBIterator(self)

	def Update_Null(self,current,rt):
		pass

	def Update(self,current,rt):

		percent=float(current)*100/self.total
		arrow='-'*int(percent/100*self.bar_length-1)+'>'
		spaces=' '*(self.bar_length-len(arrow))
		itps=current/rt
		projrem=(self.total-current)/itps

		info_string=("\r\033[32m %s [%s%s] (%d/%d) %d%%, %.2f %s, %.2f %s, %.2f %s \033[0m        \r"
			%('Progress',arrow,spaces,current-1,self.total,percent,itps,'it/s',rt,'seconds elapsed',
				projrem,'seconds remaining'))

		sys.stdout.write(info_string)
		sys.stdout.flush()

#Custom iterator for progress bar
class PBIterator():
	def __init__(self,ProgressBar):

		self.ProgressBar=ProgressBar
		self.index=0
		self.rt=0
		self.t0=time.time()

	def __next__(self):

		if self.index<len(self.ProgressBar.iterable):

			self.index+=1
			self.rt=time.time()-self.t0

			if self.index%self.ProgressBar.freq==0:
				self.ProgressBar.update(self.index,self.rt)

			return self.ProgressBar.iterable[self.index-1]

		else:

			self.index+=1
			self.rt=time.time()-self.t0

			self.ProgressBar.update(self.index,self.rt)

			if self.ProgressBar.disp:
				
				print('\n')

			raise StopIteration


def load_itineraries(day_type="weekday"):
    # Choose the filename based on the day_type parameter
    if day_type == "weekday":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
    elif day_type == "weekend":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'
    elif day_type == "all":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl'
    else:
        raise ValueError("day_type must be one of 'weekday', 'weekend', or 'all'")

    # Check if the file exists and load it
    if os.path.isfile(filename):
        itineraries = pd.read_pickle(filename)
        print(f'Loaded itineraries from {day_type} file.')
        return itineraries
    else:
        print(f'File {filename} does not exist.')
        return None


def read_pkl(file_path):
	# Load the pickle file
	with open(file_path, 'rb') as file:
		data = pickle.load(file)
	# Check if the data is an array and display or convert it accordingly
	if isinstance(data, np.ndarray):
		# If the data is a single array
		print("Data contains a single array:")
		print(data)
		# Optionally, convert to DataFrame if it's a 2D array
		if data.ndim == 2:
			df = pd.DataFrame(data)
			print("\nConverted to DataFrame:")
			print(df.head())
	elif isinstance(data, list) and all(isinstance(i, np.ndarray) for i in data):
		# If the data is a list of arrays
		print("Data contains a list of arrays:")
		for idx, array in enumerate(data):
			print(f"\nArray {idx + 1}:")
			print(array)
			# Optionally, convert each array to DataFrame if it's 2D
			if array.ndim == 2:
				df = pd.DataFrame(array)
				print("\nConverted to DataFrame:")
				print(df.head())
	else:
		print("The pickle file contains data that is not an array or list of arrays.")
	combined_df = pd.concat([entry['trips'] for entry in data], ignore_index=True)
	return combined_df
