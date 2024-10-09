import sys
import time
import argparse
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

from .utilities import ProgressBar

from numba import jit

class Vehicle():

	def __init__(self,
		itinerary,
		destination_charger_likelihood=.05,
		home_charger_likelihood=1,
		work_charger_likelihood=.5,
		en_route_charger_likelihood=1,
		consumption=478.8,
		battery_capacity=82*3.6e6,
		initial_soc=.5,
		final_soc=.5,
		payment_time=60,
		destination_charger_power=12100,
		en_route_charger_power=150000,
		home_charger_power=12100,
		work_charger_power=12100,
		ac_dc_conversion_efficiency=.88,
		max_soc=1,
		min_range=25000,
		quanta_soc=50,
		quanta_ac_charging=2,
		quanta_dc_charging=10,
		max_en_route_charging=7200,
		en_route_charging_time=15*60,
		final_soc_penalty=1e10,
		bounds_violation_penalty=1e50,
		tiles=7,
		time_multiplier=1,
		cost_multiplier=60,
		electricity_times=np.arange(0,25,1)*3600,
		electricity_rates=np.array([
			0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,
			0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,
			.123,.123,.123,.123,0.055,0.055,0.055
			])/3.6e6*10,
		low_rate_charging_multiplier=1.1, #[-]
		high_rate_charging_multiplier=1.5, #[-]
		rng_seed=0
		):

		self.initial_soc=initial_soc #[-]
		self.final_soc=final_soc #[-]
		self.payment_time=payment_time #[-]
		self.destination_charger_likelihood=destination_charger_likelihood #[-]
		self.home_charger_likelihood=home_charger_likelihood #[-]
		self.work_charger_likelihood=work_charger_likelihood #[-]
		self.en_route_charger_likelihood=en_route_charger_likelihood #[-]
		self.destination_charger_power=destination_charger_power #[W
		self.en_route_charger_power=en_route_charger_power #[W]
		self.home_charger_power=home_charger_power #[W]
		self.work_charger_power=work_charger_power #[W]
		self.consumption=consumption #[J/m]
		self.battery_capacity=battery_capacity #[J]
		self.ac_dc_conversion_efficiency=ac_dc_conversion_efficiency #[-]
		self.max_soc=max_soc #[-]
		self.min_range=min_range #[m]
		self.SOC_Min=self.min_range*self.consumption/self.battery_capacity #[-]
		self.quanta_soc=quanta_soc #[-]
		self.x=np.linspace(0,1,quanta_soc) #[-]
		self.quanta_ac_charging=quanta_ac_charging #[-]
		self.quanta_dc_charging=quanta_dc_charging #[-]
		self.u1=np.linspace(0,1,quanta_ac_charging) #[s]
		self.max_en_route_charging=max_en_route_charging #[s]
		self.u2=np.linspace(0,1,quanta_dc_charging) #[s]
		self.en_route_charging_time=en_route_charging_time #[s]
		self.final_soc_penalty=final_soc_penalty
		self.bounds_violation_penalty=bounds_violation_penalty
		self.tiles=tiles #[-]
		self.time_multiplier=time_multiplier/(time_multiplier+cost_multiplier) #[-]
		self.cost_multiplier=cost_multiplier/(time_multiplier+cost_multiplier) #[-]
		self.electricity_times=electricity_times #[-]
		self.electricity_rates=electricity_rates #[$/J]
		self.low_rate_charging_multiplier=low_rate_charging_multiplier #[-]
		self.high_rate_charging_multiplier=high_rate_charging_multiplier #[-]
		self.rng_seed=rng_seed
		
		self.itineraryArrays(itinerary)

	def itineraryArrays(self,itinerary):

		#Adding trip and dwell durations
		durations=itinerary['TRVLCMIN'].to_numpy()
		dwell_times=itinerary['DWELTIME'].to_numpy()

		#Fixing any non-real dwells
		dwell_times[dwell_times<0]=dwell_times[dwell_times>=0].mean()
		
		#Padding with overnight dwell
		sum_of_times=durations.sum()+dwell_times[:-1].sum()

		if sum_of_times>=1440:
			ratio=1440/sum_of_times
			dwell_times*=ratio
			durations*=ratio
		else:
			final_dwell=1440-durations.sum()-dwell_times[:-1].sum()
			dwell_times[-1]=final_dwell

		#Populating itinerary arrays
		self.trip_distances=np.tile(
		itinerary['TRPMILES'].to_numpy(),self.tiles)*1609.34 #[m]
		self.trip_times=np.tile(durations,self.tiles)*60 #[s]
		self.trip_mean_speeds=self.trip_distances/self.trip_times #[m/s]
		self.dwells=np.tile(dwell_times,self.tiles)*60
		self.location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),self.tiles)
		self.is_home=self.location_types==1
		self.is_work=self.location_types==10
		self.is_other=(~self.is_home&~self.is_work)

		self.destination_charger_power_array=np.array(
			[self.destination_charger_power]*len(self.dwells))
		# if self.rng_seed:
		# 	seed=self.rng_seed
		# else:
		# 	seed=np.random.randint(1e6)
		# # print(seed)
		# generator=np.random.default_rng(seed=seed)
		# charger_selection=generator.random(len(self.destination_charger_power_array))
		# no_charger=charger_selection>=self.destination_charger_likelihood
		# # print(no_charger)
		# self.destination_charger_power_array[no_charger]=0

		#Adding home chargers to home destinations
		self.destination_charger_power_array[self.is_home]=self.home_charger_power

		#Adding work chargers to work destinations
		self.destination_charger_power_array[self.is_work]=self.work_charger_power

		self.destination_charger_power_array[self.is_other]=self.destination_charger_power

		self.destination_charger_availability=np.zeros(len(self.dwells))
		self.destination_charger_availability[self.is_home]=self.home_charger_likelihood
		self.destination_charger_availability[self.is_work]=self.work_charger_likelihood
		self.destination_charger_availability[self.is_other]=(
			self.destination_charger_likelihood)

		self.en_route_charger_availability=self.en_route_charger_likelihood

		#Cost of charging
		trip_start_times=itinerary['STRTTIME'].to_numpy()*60
		trip_end_times=itinerary['ENDTIME'].to_numpy()*60
		trip_mean_times=(trip_start_times+trip_end_times)/2
		dwell_start_times=itinerary['ENDTIME'].to_numpy()*60
		dwell_end_times=itinerary['ENDTIME'].to_numpy()*60+dwell_times*60
		dwell_mean_times=(dwell_start_times+dwell_end_times)/2

		self.en_route_charge_cost_array=np.tile(
			self.high_rate_charging_multiplier*np.interp(
			trip_mean_times,self.electricity_times,self.electricity_rates),self.tiles)
		self.destination_charge_cost_array=np.tile(
			self.low_rate_charging_multiplier*np.interp(
			dwell_mean_times,self.electricity_times,self.electricity_rates),self.tiles)

	def Optimize(self):

		soc_vals=self.x
		u1_vals=self.u1
		u2_vals=self.u2
		soc_grid,u1_grid,u2_grid=np.meshgrid(soc_vals,u1_vals,u2_vals,indexing='ij')

		#Pre-calculating discharge events for each trip
		discharge_events=self.trip_distances*self.consumption/self.battery_capacity

		optimal_u1,optimal_u2,cost_to_go=OCS_Optimize(
			self.dwells,self.SOC_Min,self.max_soc,
			soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,self.max_en_route_charging,
			self.destination_charger_power_array,self.en_route_charger_power,
			self.en_route_charging_time,self.is_other,self.payment_time,
			discharge_events,self.final_soc,self.final_soc_penalty,
			self.bounds_violation_penalty,self.battery_capacity,
			self.ac_dc_conversion_efficiency,
			self.time_multiplier,self.cost_multiplier,
			self.en_route_charge_cost_array,self.destination_charge_cost_array,
			self.en_route_charger_availability,self.destination_charger_availability
			)

		optimal_strategy=[optimal_u1,optimal_u2]
		self.optimal_strategy=optimal_strategy
		self.cost_to_go=cost_to_go

		return optimal_strategy,cost_to_go

	def Evaluate(self,optimal_strategy=[]):

		# self.destination_charger_availability=np.ones_like(
		# 	self.destination_charger_availability)

		# print(self.destination_charger_availability)

		if not optimal_strategy:
			optimal_strategy=self.optimal_strategy

		soc_vals=self.x
		u1_vals=self.u1
		u2_vals=self.u2

		#Pre-calculating discharge events for each trip
		discharge_events=self.trip_distances*self.consumption/self.battery_capacity

		optimal_u1_trace,optimal_u2_trace,state_trace,cost_trace=OCS_Evaluate(
			optimal_strategy[0],optimal_strategy[1],self.initial_soc,
			self.trip_distances,self.dwells,self.max_en_route_charging,soc_vals,
			self.destination_charger_power_array,self.en_route_charger_power,
			self.en_route_charging_time,self.is_other,self.payment_time,
			discharge_events,self.battery_capacity,self.ac_dc_conversion_efficiency,
			self.en_route_charge_cost_array,self.destination_charge_cost_array,
			self.en_route_charger_availability,self.destination_charger_availability
			)

		optimal_control=[optimal_u1_trace,optimal_u2_trace]

		self.optimal_control=optimal_control
		self.state_trace=state_trace
		self.cost_trace=cost_trace

		return optimal_control,state_trace,cost_trace

	def Solve(self):

		self.Optimize()
		self.Evaluate()

		return self.optimal_strategy,self.optimal_control,self.state_trace,self.cost_trace

@jit(nopython=True,cache=True)
def OCS_Optimize(dwell_times,soc_lb,soc_ub,
	soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,u2_max,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,final_soc,final_soc_penalty,
	out_of_bounds_penalty,battery_capacity,nu_ac_dc,
	time_multiplier,cost_multiplier,
	en_route_charge_cost_array,location_charge_cost_array,
	en_route_charger_availability,location_charger_availability,
	):

	#Length of the trips vector
	n=len(dwell_times)

	#Initializing loop variables
	cost_to_go=np.empty((n,len(soc_vals)))
	cost_to_go[:]=np.nan
	optimal_u1=np.empty((n,len(soc_vals)))
	optimal_u1[:]=np.nan
	optimal_u2=np.empty((n,len(soc_vals)))
	optimal_u2[:]=np.nan

	#Main loop
	for idx in np.arange(n-1,-1,-1):

		#Initializing state and control
		soc=soc_grid.copy()
		soc_nc=soc_grid.copy()
		u1=u1_grid.copy()
		u2=u2_grid.copy()

		#Assigning charging rate for current time-step
		u1*=dwell_times[idx] #Control for location charging is the charging time
		u2*=u2_max #Control for en-route charging is charge time

		#Updating state

		soc-=discharge_events[idx]
		soc_nc-=discharge_events[idx]

		#Initializing cost array
		cost=np.zeros(soc_grid.shape)
		cost_nc=np.zeros(soc_grid.shape)

		#Applying location charging control
		if location_charge_rates[idx]>0:
			soc+=CalculateArrayCharge_AC(
					location_charge_rates[idx],soc,u1,nu_ac_dc,battery_capacity)

			if is_other[idx]:
				cost+=time_multiplier*plug_in_penalty

			for idx1 in range(soc_vals.size):
				for idx2 in range(u1_vals.size):
					for idx3 in range(u2_vals.size):
						if u1[idx1,idx2,idx3]>0:
							cost[idx1,idx2,idx3]+=cost_multiplier*(u1[idx1,idx2,idx3]*
								location_charge_rates[idx]*location_charge_cost_array[idx])

		#Applying en-route charging control
		if en_route_charge_rate>0:
			delta_soc=CalculateArrayCharge_DC(
						en_route_charge_rate,soc,u2,nu_ac_dc,battery_capacity)
			soc+=delta_soc
			soc_nc+=delta_soc

			for idx1 in range(soc_vals.size):
				for idx2 in range(u1_vals.size):
					for idx3 in range(u2_vals.size):
						if u2[idx1,idx2,idx3]>0:

							time_penalty=time_multiplier*(u2[idx1,idx2,idx3]+
								en_route_charging_penalty+plug_in_penalty)
							cost_penalty=cost_multiplier*(u2[idx1,idx2,idx3]*
								en_route_charge_rate*en_route_charge_cost_array[idx])

							cost[idx1,idx2,idx3]+=time_penalty+cost_penalty
							cost_nc[idx1,idx2,idx3]+=time_penalty+cost_penalty

		#Applying boundary costs
		for idx1 in range(soc_vals.size):
			for idx2 in range(u1_vals.size):
				for idx3 in range(u2_vals.size):
					if soc[idx1,idx2,idx3]>soc_ub:
						cost[idx1,idx2,idx3]=np.inf
					elif soc[idx1,idx2,idx3]<soc_lb:
						cost[idx1,idx2,idx3]=np.inf
					if soc_nc[idx1,idx2,idx3]>soc_ub:
						cost_nc[idx1,idx2,idx3]=np.inf
					elif soc_nc[idx1,idx2,idx3]<soc_lb:
						cost_nc[idx1,idx2,idx3]=np.inf

		if idx==n-1:
			#Applying the final-state penalty
			diff=np.abs(soc-final_soc)
			penalty=diff*final_soc_penalty
			diff_nc=np.abs(soc_nc-final_soc)
			penalty_nc=diff_nc*final_soc_penalty

			for idx1 in range(soc_vals.size):
				for idx2 in range(u1_vals.size):
					for idx3 in range(u2_vals.size):
						if diff[idx1,idx2,idx3]<0:
							penalty[idx1,idx2,idx3]=np.inf
						if diff_nc[idx1,idx2,idx3]<0:
							penalty_nc[idx1,idx2,idx3]=np.inf
			cost+=penalty
			cost_nc+=penalty_nc

		else:
			#Adding cost-to-go
			cost+=np.interp(soc,soc_vals,cost_to_go[idx+1])
			cost_nc+=np.interp(soc_nc,soc_vals,cost_to_go[idx+1])

		# print(location_charger_availability[idx])
		# location_charger_availability[idx]=.99
		cost_comb=(
			(location_charger_availability[idx]+1e-10)*cost+
			(1-location_charger_availability[idx]+1e-10)*cost_nc)
		# print(soc.mean(),soc_nc.mean(),cost.mean(),cost_nc.mean())

		#Finding optimal controls and cost-to-go - 
		#Optimal controls for each starting SOC are the controls which result in
		#the lowest cost at that SOC. Cost-to-go is the cost of the optimal
		#controls at each starting SOC
		for idx1 in range(soc_vals.size):
			mins=np.zeros(u1_vals.size) #minimum for each row
			min_inds=np.zeros(u1_vals.size) #minimum index for each row
			for idx2 in range(u1_vals.size):
				mins[idx2]=np.nanmin(cost_comb[idx1,idx2,:]) #minimum for each row
				min_inds[idx2]=np.argmin(cost_comb[idx1,idx2,:])
			min_row=np.argmin(mins) #row of minimum
			min_col=min_inds[int(min_row)] #column of minimum
			optimal_u1[idx,idx1]=u1_vals[int(min_row)]
			optimal_u2[idx,idx1]=u2_vals[int(min_col)]
			cost_to_go[idx,idx1]=cost_comb[idx1,int(min_row),int(min_col)]

	return optimal_u1,optimal_u2,cost_to_go

@jit(nopython=True,cache=True)
def OCS_Evaluate(optimal_u1,optimal_u2,initial_soc,
	trip_distances,dwell_times,u2_max,soc_vals,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,battery_capacity,nu_ac_dc,
	en_route_charge_cost_array,location_charge_cost_array,
	en_route_charger_availability,location_charger_availability,
	):

	#Length of the time vector
	n=len(dwell_times)

	#Initializing loop variables
	optimal_u1_trace=np.zeros(n+1)
	optimal_u2_trace=np.zeros(n+1)
	soc_trace=np.zeros(n+1)
	soc_trace[0]=initial_soc
	cost_trace=np.zeros((2,n+1))

	#Main loop
	soc=initial_soc
	for idx in np.arange(0,n,1):

		#Updating state
		soc-=discharge_events[idx]

		#initializig cost
		cost=0
		time=0

		#Applying location charging control
		

		delta_soc_u1=0
		optimal_u1_value=0
		available=np.random.rand()<=location_charger_availability[idx]
		# available=True
		# print(available)
		if available:
			optimal_u1_value=np.interp(soc,soc_vals,optimal_u1[idx])*dwell_times[idx]
			if optimal_u1_value>0:
				delta_soc_u1=CalculateCharge_AC(
						location_charge_rates[idx],soc,optimal_u1_value,
						nu_ac_dc,battery_capacity)

				cost+=delta_soc_u1*battery_capacity*location_charge_cost_array[idx]

				if is_other[idx]:
					time+=plug_in_penalty

		soc+=delta_soc_u1

		optimal_u2_value=np.interp(soc,soc_vals,optimal_u2[idx])*u2_max

		delta_soc_u2=0
		if optimal_u2_value>0:
			delta_soc_u2=CalculateCharge_DC(
					en_route_charge_rate,soc,optimal_u2_value,
					nu_ac_dc,battery_capacity)

			if optimal_u2_value>0:
				time+=optimal_u2_value+(
					en_route_charging_penalty+plug_in_penalty)
				cost+=delta_soc_u2*battery_capacity*en_route_charge_cost_array[idx]

		soc+=delta_soc_u2

		optimal_u1_trace[idx+1]=optimal_u1_value
		optimal_u2_trace[idx+1]=optimal_u2_value
		soc_trace[idx+1]=soc
		cost_trace[0,idx+1]=time
		cost_trace[1,idx+1]=cost

	return optimal_u1_trace,optimal_u2_trace,soc_trace,cost_trace

@jit(nopython=True,cache=True)
def CalculateCharge_DC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	P_DC=P_AC*ac_dc_conversion_efficiency #[W] DC power received from charger after
	#accounting for AC/DC converter loss
	Lambda_Charging=P_DC/battery_capacity/.2 #Exponential charging factor
	t_80=(.8-SOC)*battery_capacity/P_DC
	if td_charge<=t_80:
		Delta_SOC=P_DC/battery_capacity*td_charge
	else:
		Delta_SOC=.8-SOC+.2*(1-np.exp(-Lambda_Charging*(td_charge-t_80)))
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateArrayCharge_DC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	#Calcualting the SOC gained from a charging event of duration td_charge
	#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
	#and tails off for the last 20% approaching 100% SOC at t=infiniti
	Delta_SOC=np.zeros(SOC.shape) #Initializing the SOC delta vector
	for idx1 in range(SOC.shape[0]):
		for idx2 in range(SOC.shape[1]):
			for idx3 in range(SOC.shape[2]):
				Delta_SOC[idx1,idx2,idx3]=CalculateCharge_DC(
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],
					ac_dc_conversion_efficiency,battery_capacity)
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateCharge_AC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	# P_DC=P_AC*ac_dc_conversion_efficiency #[W] DC power received from charger after
	#accounting for AC/DC converter loss
	# t_100=(1-SOC)*battery_capacity/P_DC
	# if td_charge<=t_100:
	Delta_SOC=P_AC/battery_capacity*td_charge
	# return Delta_SOC
	# else:
		# Delta_SOC=1.-SOC
	return min([Delta_SOC,1.-SOC])

@jit(nopython=True,cache=True)
def CalculateArrayCharge_AC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	#Calcualting the SOC gained from a charging event of duration td_charge
	#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
	#and tails off for the last 20% approaching 100% SOC at t=infiniti
	Delta_SOC=np.zeros(SOC.shape) #Initializing the SOC delta vector
	for idx1 in range(SOC.shape[0]):
		for idx2 in range(SOC.shape[1]):
			for idx3 in range(SOC.shape[2]):
				Delta_SOC[idx1,idx2,idx3]=CalculateCharge_AC(
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],
					ac_dc_conversion_efficiency,battery_capacity)
	return Delta_SOC