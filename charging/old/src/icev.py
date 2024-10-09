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

@jit(nopython=True,cache=True)
def Interp2D(x,y,xg,yg,zg):
	'''
	x - value of x
	y - value of y
	xg - meshgrid [[x]*n]
	yg - meshgrid [n*[y]]
	zg - values of z corresponding to grids
	Example:
	xg=array([
			[0, 0, 0],
			[1, 1, 1],
			[2, 2, 2]
			]),
	yg=array([
			[0, 1, 2],
			[0, 1, 2],
			[0, 1, 2]
			]),
	zg=array([
			[1, 4, 0],
			[0, 7, 3],
			[9, 9, 4]
			]))
	x,y=1.3,.5
	Interp2D(x,y,xg,yg,zg) -> 
	'''
	zv=np.zeros(xg.shape[0])
	for idx in range(xg.shape[0]):
		zv[idx]=np.interp(y,yg[idx,:],zg[idx,:])
	return np.interp(x,xg[:,0],zv)

class PHEV():

	def __init__(self,
		itinerary,
		charger_likelihood=.1,
		consumption_cd=479,
		consumption_cs=2355,
		battery_capacity=82*1000*3600,
		fuel_tank_capacity=300*1000*3600,
		initial_soc=.5,
		initial_sof=.5,
		final_soc=.5,
		final_sof=.5,
		payment_time=60,
		feuling_power=121300000*7/60,
		charger_power=12100,
		home_charger_power=12100,
		work_charger_power=12100,
		ac_dc_conversion_efficiency=.88,
		max_soc=1,
		min_soc=.2,
		min_range=25000,
		quanta_soc=10,
		quanta_sof=10,
		quanta_charging=5,
		quanta_fueling=2,
		fueling_travel_time=5*60,
		final_soc_penalty=1e10,
		final_sof_penalty=1e10,
		bounds_violation_penalty=1e10,
		tiles=7,
		time_multiplier=1,
		cost_multiplier=60,
		electricity_times=np.arange(0,25,1)*3600,
		electricity_rates=np.array([
			0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,
			0.11,0.11,0.11,0.11,0.20,0.20,0.28,0.28,0.28,
			0.28,0.28,0.11,0.11,0.11,0.11,0.11
			])/3.6e6, #[$]
		fuel_price=3.5/33.7/3.6e6, #[$/J]
		charging_premium=0.1, #[-]
		rng_seed=0
		):

		self.initial_soc=initial_soc #[-]
		self.final_soc=final_soc #[-]
		self.initial_sof=initial_sof #[-]
		self.final_sof=final_sof #[-]
		self.payment_time=payment_time #[-]
		self.charger_likelihood=charger_likelihood #[-]
		self.charger_power=charger_power #[W
		self.home_charger_power=home_charger_power #[W]
		self.work_charger_power=work_charger_power #[W]
		self.consumption_cd=consumption_cd #[J/m]
		self.consumption_cs=consumption_cs #[J/m]
		self.battery_capacity=battery_capacity #[J]
		self.fuel_tank_capacity=fuel_tank_capacity #[J]
		self.ac_dc_conversion_efficiency=ac_dc_conversion_efficiency #[-]
		self.max_soc=max_soc #[-]
		self.min_soc=min_soc #[-]
		self.min_range=min_range #[m]
		self.min_sof=self.min_range*self.consumption_cs/self.fuel_tank_capacity #[-]
		self.quanta_soc=quanta_soc #[-]
		self.soc=np.linspace(0,1,quanta_soc) #[-]
		self.quanta_sof=quanta_sof #[-]
		self.sof=np.linspace(0,1,quanta_sof) #[-]
		self.quanta_charging=quanta_charging #[-]
		self.u1=np.linspace(0,1,quanta_charging) #[s]
		self.quanta_fueling=quanta_fueling #[-]
		self.u2=np.linspace(0,1,quanta_fueling) #[s]
		self.fueling_travel_time=fueling_travel_time #[s]
		self.final_soc_penalty=final_soc_penalty
		self.final_sof_penalty=final_sof_penalty
		self.bounds_violation_penalty=bounds_violation_penalty
		self.tiles=tiles #[-]
		self.time_multiplier=time_multiplier #[-]
		self.cost_multiplier=cost_multiplier #[-]
		self.electricity_times=electricity_times #[-]
		self.electricity_rates=electricity_rates #[$/J]
		self.charging_premium=charging_premium #[-]
		self.feuling_power=feuling_power
		self.fuel_price=fuel_price
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
		self.trip_distances=np.tile(itinerary['TRPMILES'].to_numpy(),self.tiles)*1609.34 #[m]
		self.trip_times=np.tile(durations,self.tiles)*60 #[s]
		self.trip_mean_speeds=self.trip_distances/self.trip_times #[m/s]
		self.dwells=np.tile(dwell_times,self.tiles)*60
		self.location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),self.tiles)
		self.is_home=self.location_types==1
		self.is_work=self.location_types==10
		self.is_other=(~self.is_home&~self.is_work)

		self.charger_power_array=np.array([self.charger_power]*len(self.dwells))
		if self.rng_seed:
			seed=self.rng_seed
		else:
			seed=np.random.randint(1e6)
		# print(seed)
		generator=np.random.default_rng(seed=seed)
		charger_selection=generator.random(len(self.charger_power_array))
		no_charger=charger_selection>=self.charger_likelihood
		# print(no_charger)
		self.charger_power_array[no_charger]=0

		#Adding home chargers to home destinations
		self.charger_power_array[self.is_home]=self.home_charger_power

		#Adding work chargers to work destinations
		self.charger_power_array[self.is_work]=self.work_charger_power

		# print(self.destination_charger_power_array)

		#Cost of charging
		dwell_start_times=itinerary['ENDTIME'].to_numpy()*60
		dwell_end_times=itinerary['ENDTIME'].to_numpy()*60+dwell_times*60
		dwell_mean_times=(dwell_start_times+dwell_end_times)/2

		self.charge_cost_array=np.tile(self.charging_premium*np.interp(
			dwell_mean_times,self.electricity_times,self.electricity_rates),self.tiles)

		# print(self.destination_charge_cost_array)

	def Optimize(self):

		x1_vals=self.soc
		x2_vals=self.sof
		u1_vals=self.u1
		u2_vals=self.u2

		x1_grid,x2_grid,u1_grid,u2_grid=np.meshgrid(
			x1_vals,x2_vals,u1_vals,u2_vals,indexing='ij')

		#Interpolation matrices
		x1_interp_grid,x2_interp_grid=np.meshgrid(x1_vals,x2_vals,indexing='ij')

		optimal_u1,optimal_u2,cost_to_go=PHEV_Optimize(

			self.dwells,self.trip_distances,
			self.charger_power_array,self.charge_cost_array,
			self.fuel_price,self.is_other,self.payment_time,self.fueling_travel_time,
			self.battery_capacity,self.ac_dc_conversion_efficiency,
			self.fuel_tank_capacity,self.feuling_power,
			self.min_soc,self.max_soc,self.min_sof,1,
			x1_vals,x1_grid,x2_vals,x2_grid,
			u1_vals,u1_grid,u2_vals,u2_grid,
			self.final_soc,self.final_soc_penalty,
			self.final_sof,self.final_sof_penalty,
			self.bounds_violation_penalty,
			self.time_multiplier,self.cost_multiplier,
			self.consumption_cd,self.consumption_cs,
			x1_interp_grid,x2_interp_grid)

		self.optimal_control=[optimal_u1,optimal_u2]
		self.cost_to_go=cost_to_go

		return [optimal_u1,optimal_u2],cost_to_go

	def Evaluate(self,optimal_control=[]):

		if optimal_control:
			self.optimal_control=optimal_control

		x1_vals=self.soc
		x2_vals=self.sof
		u1_vals=self.u1
		u2_vals=self.u2

		#Interpolation matrices
		x1_interp_grid,x2_interp_grid=np.meshgrid(x1_vals,x2_vals,indexing='ij')


		(optimal_u1_trace,optimal_u2_trace,x1_trace,x2_trace,energizing_cost,
			energizing_sic,electric_distance,conventional_distance)=PHEV_Evaluate(
			self.trip_distances,self.dwells,
			self.optimal_control[0],self.optimal_control[1],
			self.initial_soc,self.initial_sof,
			x1_interp_grid,x2_interp_grid,
			self.min_soc,self.max_soc,self.min_sof,1,
			self.charger_power_array,self.charge_cost_array,
			self.is_other,self.payment_time,
			self.battery_capacity,self.ac_dc_conversion_efficiency,
			self.fuel_tank_capacity,self.feuling_power,
			self.fuel_price,self.fueling_travel_time,
			self.consumption_cd,self.consumption_cs)

		self.optimal_trace=[optimal_u1_trace,optimal_u2_trace]

		return ([optimal_u1_trace,optimal_u2_trace],x1_trace,x2_trace,
			energizing_cost,energizing_sic,electric_distance,conventional_distance)

@jit(nopython=True,cache=True)
def PHEV_Optimize(
	dwell_times,trip_distances,
	charge_rates,charge_cost_array,fuel_cost,
	is_other,plug_in_penalty,fueling_travel_penalty,
	battery_capacity,nu_ac_dc,
	fuel_tank_capacity,fueling_rate,
	x1_lb,x1_ub,x2_lb,x2_ub,
	x1_vals,x1_grid,x2_vals,x2_grid,
	u1_vals,u1_grid,u2_vals,u2_grid,
	final_x1,final_x1_penalty,
	final_x2,final_x2_penalty,
	out_of_bounds_penalty,
	time_multiplier,cost_multiplier,
	consumption_cd,consumption_cs,
	x1_interp_grid,x2_interp_grid
):

	#Length of the trips vector
	n=len(dwell_times)

	#Initializing loop variables
	cost_to_go=np.empty((n,x1_vals.size,x2_vals.size))
	cost_to_go[:]=np.nan
	optimal_u1=np.empty((n,x1_vals.size,x2_vals.size))
	optimal_u1[:]=np.nan
	optimal_u2=np.empty((n,x1_vals.size,x2_vals.size))
	optimal_u2[:]=np.nan

	#Main loop
	for idx in np.arange(n-1,-1,-1):

		#Initializing state and control
		x1=x1_grid.copy()
		x2=x2_grid.copy()
		u1=u1_grid.copy()
		u2=u2_grid.copy()

		#Assigning charging rate for current time-step
		u1*=dwell_times[idx] #Control for location charging is the charging time

		#Initializing cost array
		cost=np.zeros(x1_grid.shape)

		#Trip energy consumption - will always draw from battery first
		trip_energy_consumption=trip_distances[idx]*consumption_cd

		for idx1 in range(x1_vals.size):
			for idx2 in range(x2_vals.size):
				lowest_cost=np.inf
				lowest_cost_idx3=0
				lowest_cost_idx4=0

				for idx3 in range(u1_vals.size):
					for idx4 in range(u2_vals.size):

						#Updating state
						if trip_energy_consumption<=(x1[idx1,idx2,idx3,idx4]-x1_lb)*battery_capacity:
							#If the trip energy_consumption is less than energy left in the battery
							#then the entire trip depletes the battery
							x1[idx1,idx2,idx3,idx4]-=trip_energy_consumption/battery_capacity
						else:
							#Otherwise the battery is depleted to minimum and the rest of the trip
							#is on engine power
							trip_energy_consumption_cs=(
								trip_energy_consumption-(x1[idx1,idx2,idx3,idx4]-x1_lb)*battery_capacity)*(
								consumption_cs/consumption_cd)
							x1[idx1,idx2,idx3,idx4]=min([x1[idx1,idx2,idx3,idx4],x1_lb])
							x2[idx1,idx2,idx3,idx4]-=trip_energy_consumption_cs/fuel_tank_capacity

						#Applying charging control
						if charge_rates[idx]>0:
							delta_x1=CalculateCharge_AC(
								charge_rates[idx],x1[idx1,idx2,idx3,idx4],u1[idx1,idx2,idx3,idx4],
								nu_ac_dc,battery_capacity)

							charge_energy=delta_x1*battery_capacity
							charge_cost=charge_energy*charge_cost_array[idx]
							cost[idx1,idx2,idx3,idx4]+=cost_multiplier*charge_cost

							#Only add the time penalty for payment if not at home or work
							if is_other[idx]:
								cost[idx1,idx2,idx3,idx4]+=time_multiplier*plug_in_penalty

							x1[idx1,idx2,idx3,idx4]+=delta_x1

						#Applying fueling control
						if u2[idx1,idx2,idx3,idx4]>0:
							fueling_energy=(1-x2[idx1,idx2,idx3,idx4])*fuel_tank_capacity
							fueling_time=fueling_energy/fueling_rate
							
							cost[idx1,idx2,idx3,idx4]+=time_multiplier*(
								fueling_time+fueling_travel_penalty)
							cost[idx1,idx2,idx3,idx4]+=cost_multiplier*(
								fueling_energy*fuel_cost)

							x2[idx1,idx2,idx3,idx4]=1

						# print(cost[idx1,idx2,idx3,idx4])
						#Applying boundary costs
						if x1[idx1,idx2,idx3,idx4]>x1_ub:
							cost[idx1,idx2,idx3,idx4]+=(
								x1[idx1,idx2,idx3,idx4]-x1_ub)**2*out_of_bounds_penalty
						if x1[idx1,idx2,idx3,idx4]<x1_lb:
							# print('a')
							cost[idx1,idx2,idx3,idx4]+=(
								x1[idx1,idx2,idx3,idx4]-x1_lb)**2*out_of_bounds_penalty
						if x2[idx1,idx2,idx3,idx4]>x2_ub:
							cost[idx1,idx2,idx3,idx4]+=(
								x2[idx1,idx2,idx3,idx4]-x2_ub)**2*out_of_bounds_penalty
						if x2[idx1,idx2,idx3,idx4]<x2_lb:
							# print('b')
							cost[idx1,idx2,idx3,idx4]+=(
								x2[idx1,idx2,idx3,idx4]-x2_lb)**2*out_of_bounds_penalty
						# print(cost[idx1,idx2,idx3,idx4])

						#Cost-to-go
						if idx==n-1:
							#Applying the final-state penalty
							diff_x1=x1[idx1,idx2,idx3,idx4]-final_x1
							penalty_x1=final_x1_penalty*(diff_x1)**2*final_x1_penalty
							diff_x2=x2[idx1,idx2,idx3,idx4]-final_x2
							penalty_x2=final_x2_penalty*(diff_x2)**2*final_x2_penalty

							if diff_x1>0:
								penalty_x1=0
							if diff_x2>0:
								penalty_x2=0

							cost[idx1,idx2,idx3,idx4]+=penalty_x1+penalty_x2
						else:
							#Adding cost-to-go
							ctg=Interp2D(x1[idx1,idx2,idx3,idx4],x2[idx1,idx2,idx3,idx4],
								x1_interp_grid,x2_interp_grid,cost_to_go[idx+1])

							# if idx==n-2:
								# print(ctg)

							cost[idx1,idx2,idx3,idx4]+=ctg

						#Updating optimal controls and cost-to-go
						if cost[idx1,idx2,idx3,idx4]<lowest_cost:
							# print(idx,lowest_cost,cost[idx1,idx2,idx3,idx4])
							lowest_cost=cost[idx1,idx2,idx3,idx4]
							lowest_cost_idx3=idx3
							lowest_cost_idx4=idx4

				#Assigning optimal controls and cost-to-go
				cost_to_go[idx,idx1,idx2]=lowest_cost
				optimal_u1[idx,idx1,idx2]=u1_vals[lowest_cost_idx3]
				optimal_u2[idx,idx1,idx2]=u2_vals[lowest_cost_idx4]
		# print(cost_to_go[idx])



	return optimal_u1,optimal_u2,cost_to_go

@jit(nopython=True,cache=True)
def PHEV_Evaluate(
	trip_distances,dwell_times,
	optimal_u1,optimal_u2,
	initial_x1,initial_x2,
	x1_grid,x2_grid,
	x1_lb,x1_ub,x2_lb,x2_ub,
	charge_rates,charge_cost_array,
	is_other,plug_in_penalty,
	battery_capacity,nu_ac_dc,
	fuel_tank_capacity,fueling_rate,
	fuel_cost,fueling_time_penalty,
	consumption_cd,consumption_cs
):

	#Length of the time vector
	n=len(dwell_times)

	#Initializing loop variables
	optimal_u1_trace=np.empty(n)
	optimal_u2_trace=np.empty(n)

	x1_trace=np.empty(n+1)
	x1_trace[0]=initial_x1

	x2_trace=np.empty(n+1)
	x2_trace[0]=initial_x2

	x1=initial_x1
	x2=initial_x2

	energizing_sic=0
	energizing_cost=0
	electric_distance=0
	conventional_distance=0

	#Main loop
	for idx in np.arange(0,n,1):

		#Trip energy consumption - will always draw from battery first
		trip_energy_consumption=trip_distances[idx]*consumption_cd

		#Updating state
		if trip_energy_consumption<=(x1-x1_lb)*battery_capacity:
			#If the trip energy_consumption is less than energy left in the battery
			#then the entire trip depletes the battery
			x1-=trip_energy_consumption/battery_capacity
			electric_distance+=trip_distances[idx]
		else:
			#Otherwise the battery is depleted to minimum and the rest of the trip
			#is on engine power
			trip_energy_consumption_cs=(
				trip_energy_consumption-(x1-x1_lb)*battery_capacity)*(consumption_cs/consumption_cd)
			x1=min([x1,x1_lb])
			x2-=trip_energy_consumption_cs/fuel_tank_capacity

			incremental_conventional_distance=trip_energy_consumption_cs/(
				trip_distances[idx]*consumption_cs)*trip_distances[idx]
			conventional_distance+=incremental_conventional_distance
			electric_distance+=trip_distances[idx]-incremental_conventional_distance

		#Applying charging control
		optimal_u1_trace[idx]=Interp2D(x1,x2,x1_grid,x2_grid,optimal_u1[idx])*dwell_times[idx]

		if charge_rates[idx]>0:

			delta_x1=CalculateCharge_AC(
				charge_rates[idx],x1,optimal_u1_trace[idx],nu_ac_dc,battery_capacity)

			charge_energy=delta_x1*battery_capacity
			charge_cost=charge_energy*charge_cost_array[idx]
			energizing_cost+=charge_cost

			#Only add the time penalty for payment if not at home or work
			if is_other[idx]:
				energizing_sic+=plug_in_penalty

			#updating x1
			x1+=delta_x1

		#Applying fueling control
		optimal_u2_trace[idx]=np.round(Interp2D(x1,x2,x1_grid,x2_grid,optimal_u2[idx]))

		if optimal_u2_trace[idx]>0:

			fueling_energy=(1-x2)*fuel_tank_capacity
			fueling_time=fueling_energy/fueling_rate
			fueling_cost=fueling_energy*fuel_cost

			energizing_cost+=fueling_cost
			energizing_sic+=fueling_time+fueling_time_penalty

			#updating x1
			x2=1

		x1_trace[idx+1]=x1
		x2_trace[idx+1]=x2

	energizing_sic=(energizing_sic/60)/(trip_distances.sum()/1000)

	return (optimal_u1_trace,optimal_u2_trace,x1_trace,x2_trace,
		energizing_cost,energizing_sic,electric_distance,conventional_distance)

class BEV():

	def __init__(self,
		itinerary,
		destination_charger_likelihood=.1,
		consumption=478.8,
		battery_capacity=82*1000*3600,
		starting_soc=.5,
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
			1.23,1.23,1.23,1.23,0.055,0.055,0.055
			])/3.6e6,
		low_rate_charging_premium=0.1, #[-]
		high_rate_charging_premium=0.25, #[-]
		rng_seed=0
		):

		self.starting_soc=starting_soc #[-]
		self.final_soc=final_soc #[-]
		self.payment_time=payment_time #[-]
		self.destination_charger_likelihood=destination_charger_likelihood #[-]
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
		self.time_multiplier=time_multiplier #[-]
		self.cost_multiplier=cost_multiplier #[-]
		self.electricity_times=electricity_times #[-]
		self.electricity_rates=electricity_rates #[$/J]
		self.low_rate_charging_premium=low_rate_charging_premium #[-]
		self.high_rate_charging_premium=high_rate_charging_premium #[-]
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
		self.trip_distances=np.tile(itinerary['TRPMILES'].to_numpy(),self.tiles)*1609.34 #[m]
		self.trip_times=np.tile(durations,self.tiles)*60 #[s]
		self.trip_mean_speeds=self.trip_distances/self.trip_times #[m/s]
		self.dwells=np.tile(dwell_times,self.tiles)*60
		self.location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),self.tiles)
		self.is_home=self.location_types==1
		self.is_work=self.location_types==10
		self.is_other=(~self.is_home&~self.is_work)

		self.destination_charger_power_array=np.array([self.destination_charger_power]*len(self.dwells))
		if self.rng_seed:
			seed=self.rng_seed
		else:
			seed=np.random.randint(1e6)
		# print(seed)
		generator=np.random.default_rng(seed=seed)
		charger_selection=generator.random(len(self.destination_charger_power_array))
		no_charger=charger_selection>=self.destination_charger_likelihood
		# print(no_charger)
		self.destination_charger_power_array[no_charger]=0

		#Adding home chargers to home destinations
		self.destination_charger_power_array[self.is_home]=self.home_charger_power

		#Adding work chargers to work destinations
		self.destination_charger_power_array[self.is_work]=self.work_charger_power

		# print(self.destination_charger_power_array)

		#Cost of charging
		trip_start_times=itinerary['STRTTIME'].to_numpy()*60
		trip_end_times=itinerary['ENDTIME'].to_numpy()*60
		trip_mean_times=(trip_start_times+trip_end_times)/2
		dwell_start_times=itinerary['ENDTIME'].to_numpy()*60
		dwell_end_times=itinerary['ENDTIME'].to_numpy()*60+dwell_times*60
		dwell_mean_times=(dwell_start_times+dwell_end_times)/2

		self.en_route_charge_cost_array=np.tile(self.high_rate_charging_premium*np.interp(
			trip_mean_times,self.electricity_times,self.electricity_rates),self.tiles)
		self.destination_charge_cost_array=np.tile(self.low_rate_charging_premium*np.interp(
			dwell_mean_times,self.electricity_times,self.electricity_rates),self.tiles)

		# print(self.destination_charge_cost_array)


	def Optimize(self):

		soc_vals=self.x
		u1_vals=self.u1
		u2_vals=self.u2
		soc_grid,u1_grid,u2_grid=np.meshgrid(soc_vals,u1_vals,u2_vals,indexing='ij')

		#Pre-calculating discharge events for each trip
		discharge_events=self.trip_distances*self.consumption/self.battery_capacity

		optimal_u1,optimal_u2,cost_to_go=BEV_Optimize(
			self.dwells,self.SOC_Min,self.max_soc,
			soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,self.max_en_route_charging,
			self.destination_charger_power_array,self.en_route_charger_power,
			self.en_route_charging_time,self.is_other,self.payment_time,
			discharge_events,self.final_soc,self.final_soc_penalty,
			self.bounds_violation_penalty,self.battery_capacity,self.ac_dc_conversion_efficiency,
			self.time_multiplier,self.cost_multiplier,
			self.en_route_charge_cost_array,self.destination_charge_cost_array)

		self.optimal_control=[optimal_u1,optimal_u2]
		self.cost_to_go=cost_to_go

		return [optimal_u1,optimal_u2],cost_to_go

	def Evaluate(self,optimal_control=[]):

		if optimal_control:
			self.optimal_control=optimal_control

		soc_vals=self.x
		u1_vals=self.u1
		u2_vals=self.u2

		#Pre-calculating discharge events for each trip
		discharge_events=self.trip_distances*self.consumption/self.battery_capacity

		optimal_u1_trace,optimal_u2_trace,soc_trace,sic=BEV_Evaluate(
			self.optimal_control[0],self.optimal_control[1],self.starting_soc,
			self.trip_distances,self.dwells,self.max_en_route_charging,soc_vals,
			self.destination_charger_power_array,self.en_route_charger_power,
			self.en_route_charging_time,self.is_other,self.payment_time,
			discharge_events,self.battery_capacity,self.ac_dc_conversion_efficiency)

		self.optimal_trace=[optimal_u1_trace,optimal_u2_trace]

		return [optimal_u1_trace,optimal_u2_trace],soc_trace,sic

@jit(nopython=True,cache=True)
def BEV_Optimize(dwell_times,soc_lb,soc_ub,
	soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,u2_max,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,final_soc,final_soc_penalty,
	out_of_bounds_penalty,battery_capacity,nu_ac_dc,
	time_multiplier,cost_multiplier,
	en_route_charge_cost_array,location_charge_cost_array
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
		u1=u1_grid.copy()
		u2=u2_grid.copy()

		#Assigning charging rate for current time-step
		u1*=dwell_times[idx] #Control for location charging is the charging time
		u2*=u2_max #Control for en-route charging is charge time

		#Updating state
		soc-=discharge_events[idx]

		#Initializing cost array
		cost=np.zeros(soc_grid.shape)

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
							cost[idx1,idx2,idx3]+=cost_multiplier*u1[idx1,idx2,idx3]*(
								location_charge_rates[idx]*location_charge_cost_array[idx])

		#Applying en-route charging control
		if en_route_charge_rate>0:
			soc+=CalculateArrayCharge_DC(
						en_route_charge_rate,soc,u2,nu_ac_dc,battery_capacity)
		for idx1 in range(soc_vals.size):
			for idx2 in range(u1_vals.size):
				for idx3 in range(u2_vals.size):
					if u2[idx1,idx2,idx3]>0:
						cost[idx1,idx2,idx3]+=time_multiplier*u2[idx1,idx2,idx3]+(
							en_route_charging_penalty+
							plug_in_penalty)
						cost[idx1,idx2,idx3]+=cost_multiplier*u2[idx1,idx2,idx3]*(
							en_route_charge_rate*en_route_charge_cost_array[idx])

		#Applying boundary costs
		for idx1 in range(soc_vals.size):
			for idx2 in range(u1_vals.size):
				for idx3 in range(u2_vals.size):
					if soc[idx1,idx2,idx3]>soc_ub:
						cost[idx1,idx2,idx3]+=out_of_bounds_penalty
					elif soc[idx1,idx2,idx3]<soc_lb:
						cost[idx1,idx2,idx3]+=out_of_bounds_penalty

		if idx==n-1:
			#Applying the final-state penalty
			diff=soc-final_soc
			penalty=final_soc_penalty*(diff)**2*final_soc_penalty
			# penalty[diff>0]=0
			for idx1 in range(soc_vals.size):
				for idx2 in range(u1_vals.size):
					for idx3 in range(u2_vals.size):
						if diff[idx1,idx2,idx3]>0:
							penalty[idx1,idx2,idx3]=0
			cost+=penalty
		else:
			#Adding cost-to-go
			cost+=np.interp(soc,soc_vals,cost_to_go[idx+1])

		#Finding optimal controls and cost-to-go - Optimal controls for each starting SOC are the controls which result in
		#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
		for idx1 in range(soc_vals.size):
			mins=np.zeros(u1_vals.size) #minimum for each row
			min_inds=np.zeros(u1_vals.size) #minimum index for each row
			for idx2 in range(u1_vals.size):
				mins[idx2]=np.min(cost[idx1,idx2,:]) #minimum for each row
				min_inds[idx2]=np.argmin(cost[idx1,idx2,:])
			min_row=np.argmin(mins) #row of minimum
			min_col=min_inds[int(min_row)] #column of minimum
			optimal_u1[idx,idx1]=u1_vals[int(min_row)]
			optimal_u2[idx,idx1]=u2_vals[int(min_col)]
			cost_to_go[idx,idx1]=cost[idx1,int(min_row),int(min_col)]

	return optimal_u1,optimal_u2,cost_to_go

@jit(nopython=True,cache=True)
def BEV_Evaluate(optimal_u1,optimal_u2,initial_soc,
	trip_distances,dwell_times,u2_max,soc_vals,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,battery_capacity,nu_ac_dc
):

	#Length of the time vector
	n=len(dwell_times)

	#Initializing loop variables
	optimal_u1_trace=np.empty(n)
	optimal_u2_trace=np.empty(n)

	soc_trace=np.empty(n+1)
	soc_trace[0]=initial_soc

	soc=initial_soc
	dedicated_energizing_time=0
	

	#Main loop
	soc=initial_soc
	for idx in np.arange(0,n,1):

		#Updating state
		soc-=discharge_events[idx]

		#Applying location charging control
		optimal_u1_trace[idx]=np.interp(soc,soc_vals,optimal_u1[idx])*dwell_times[idx]

		if location_charge_rates[idx]>0:
			if optimal_u1_trace[idx]>0:
				soc+=CalculateCharge_AC(
						location_charge_rates[idx],soc,optimal_u1_trace[idx],
						nu_ac_dc,battery_capacity)
				if is_other[idx]:
					dedicated_energizing_time+=plug_in_penalty


		#Applying en-route charging control
		optimal_u2_trace[idx]=np.interp(soc,soc_vals,optimal_u2[idx])*u2_max

		if en_route_charge_rate>0:
			soc+=CalculateCharge_DC(
						en_route_charge_rate,soc,optimal_u2_trace[idx],nu_ac_dc,battery_capacity)
			if optimal_u2_trace[idx]>0:
				dedicated_energizing_time+=optimal_u2_trace[idx]+(en_route_charging_penalty+
					plug_in_penalty)

		soc_trace[idx+1]=soc

	sicd=(dedicated_energizing_time/60)/(trip_distances.sum()/1000)

	return optimal_u1_trace,optimal_u2_trace,soc_trace,sicd

@jit(nopython=True,cache=True)
def CalculateCharge_DC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	P_DC=P_AC*ac_dc_conversion_efficiency #[W] DC power received from charger after accounting for AC/DC converter loss
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
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],ac_dc_conversion_efficiency,battery_capacity)
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateCharge_AC(P_AC,SOC,td_charge,ac_dc_conversion_efficiency,battery_capacity):
	P_DC=P_AC*ac_dc_conversion_efficiency #[W] DC power received from charger after accounting for AC/DC converter loss
	t_100=(1-SOC)*battery_capacity/P_DC
	if td_charge<=t_100:
		Delta_SOC=P_DC/battery_capacity*td_charge
	else:
		Delta_SOC=1.-SOC
	return Delta_SOC

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
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],ac_dc_conversion_efficiency,battery_capacity)
	return Delta_SOC

class ICV():

	def __init__(self,veh,
				consumption_city=2599.481,
				consumption_mixed=2355.779,
				consumption_highway=2094.026,
				Fuel_Tank_Capacity=528*1000*3600,
				Starting_SOF=.5,
				Final_SOF=.5,
				SOF_Max=1,
				min_range=25000,
				SOF_Quanta=20,
				speed_thresholds=np.array([35,65])*0.44704,
				Fueling_Time_Penalty=300,
				Fueling_Rate=121300000*7/60):
		
		self.Starting_SOF=Starting_SOF #[dim] ICV's SOF at start of itinerary
		self.Final_SOF=Final_SOF #[dim] ICV's SOF at end of itinerary
		self.speed_thresholds=speed_thresholds #[m/s] Upper average speed thresholds for city and mixed driving
		self.consumption_city=consumption_city #[J/m] Fuel consumption at urban speeds
		self.consumption_mixed=consumption_mixed #[J/m] Fuel consumption at medium speeds
		self.consumption_highway=consumption_highway #[J/m] Fuel consumption at highway speeds
		self.Fuel_Tank_Capacity=Fuel_Tank_Capacity #[J] Maximum energy which can be stored in the fuel tank
		self.SOF_Max=SOF_Max #[dim] Maximum allowable SOF
		self.min_range=min_range #[m] Lowest allowable remaining range
		self.SOF_Min=self.min_range*self.consumption_mixed/self.Fuel_Tank_Capacity #[dim] Lowest allowable SOF
		self.x=np.linspace(0,self.SOF_Max,SOF_Quanta) #[dim] (n_x,) array of discreet SOF values for optimization
		self.U=np.array([0,1]) #[s] (n_U,) array of discreet SOF deltas for optimization
		self.Fueling_Time_Penalty=Fueling_Time_Penalty #[s] time penalty for traveling to and re-fueling at a re-feuling station
		self.Fueling_Rate=Fueling_Rate #[W] rate of energization while fueling
		
		self.itineraryArrays(veh)
	
	def itineraryArrays(self,veh):
		#Populates itinerary arrays
		vdf=veh.df.head(100) #Need a better way to down-select the itinerary
		self.trip_distances=vdf['tripDist_mi'].to_numpy()*1609.34 #[m] Distances of trips preceeding parks
		self.trip_times=vdf['tripTime'].to_numpy() #[s] Durations of trips preceeding parks
		self.trip_mean_speeds=self.trip_distances/self.trip_times #[m/s] Speeds of trips preceeding parks
		self.dwells=vdf['dwellTime'].to_numpy() #[s] Durations of parks
	
	def Optimize(self):
		#the optimize step is the first step in the DP solver in which optimal control matrices are created. The optimize step involves
		#backwards iteration through the exogenous input tracres while the optimal control matrices are populated.
		N=len(self.dwells) #Length of itinerary
		#Initializing loop variables
		Cost_to_Go=np.empty((N,len(self.x)))
		Cost_to_Go[:]=np.nan
		Optimal_U=np.empty((N,len(self.x)))
		Optimal_U[:]=np.nan
		#Pre-calculating discharge events for each trip
		FCRate=np.ones(len(self.trip_distances))*self.consumption_mixed
		FCRate[self.trip_mean_speeds<self.speed_thresholds[0]]=self.consumption_city
		FCRate[self.trip_mean_speeds>=self.speed_thresholds[1]]=self.consumption_highway
		Trip_SOF_Deltas=self.trip_distances*FCRate/self.Fuel_Tank_Capacity
		#Main loop (backwards iteration)
		for k in np.arange(N-1,-1,-1):
			#Initializing state and controls arrays
			SOF,Fuel=np.meshgrid(self.x,self.U,indexing='ij') #(n_x,n_U) arrays of values for state and control
			#every combination of state and control is evaluated
			# print(SOF,Fuel)
			#Discharging
			SOF-=Trip_SOF_Deltas[k] #Applying discharge events to the SOF
			
			#Initializing cost array
			Cost=np.zeros((len(self.x),len(self.U))) #Array of same dimensions as SOF/Fuel which will store the
			#cost of each combination of state and control
			
			#ICVs can only fuel en-route.
			SOF[Fuel==1]=self.SOF_Max
			Cost+=((Fuel>0)*self.Fueling_Time_Penalty)
			
			#Enforcing constraints - disallowable combinations of state and controls are assigned huge costs - these will later
			#be used to identify which states are disallowable. Common practice is to assign NaN of Inf cost to disallowable
			#combinations here but this leads to complications later in the code so a very high number works better
			Cost[SOF<self.SOF_Min]=1e50 #SOF too low
			Cost[SOF>self.SOF_Max]=1e50 #SOF too high
			
			#Penalty and Cost to Go - the penalty for failing to meet the final SOF constraint is applied at step N-1
			#(the last step in the itinerary but the first processed in the loop). for all other stpes the cost-to-go is applied.
			#Cost-to-go is the cost to go from step k+1 to step k+2 (k is the current step). Cost-to-go is the "memory" element
			#of the method which ties the steps together
			if k==N-1:
				#Assinging penalty for failing to meet the final SOF constraints
				diff=self.Final_SOF-SOF
				Penalty=diff**2*1e10
				Penalty[diff<0]=0
				# print(Penalty)
				Cost+=Penalty
			else:
				#Assigning the cost-to-go
				Cost+=np.interp(SOF,self.x,Cost_to_Go[k+1])
			 
			#Finding optimal controls and cost-to-go - Optimal controls for each starting SOF are the controls which result in
			#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
			mins=np.min(Cost,axis=1) #minimum for each row
			
			min_inds=np.argmin(Cost,axis=1) #minimum axis for each row
			Optimal_U[k]=self.U[min_inds]
			Cost_to_Go[k]=mins
			
		#Identifying disallowable optimal controls - optimal controls resulting from disallowable combinations are set to -1
		#so that they can be easily filtered out (all others will be >=0). Disallowable combinations can be "optimal" if all
		#combinations are disallowable for a given SOF and step
		Optimal_U[Cost_to_Go>=1e50]=-1
		
		#Outputs of BEV.Optimize() are a list of the optimal control matrices for each control and the cost-to-go matrix
		return Optimal_U,Cost_to_Go
	
	def Evaluate(self,Optimal_Control):
		#The evaluate step is the second step in the DP method in which optimal controls and states are found for each
		#time-step in the itinerary. This is accomplished through forward iteration. Unlike the optimize step which 
		#considers all possible SOF values, the evaluate step follows only one as it iterates forward using the optimal
		#controls for the current SOF and step.
		#Initializations
		N=len(self.dwells)
		Optimal_U=np.empty(N)
		Optimal_U[:]=np.nan
		Fueling_Time_Penalty=np.zeros(N)
		SOF_Trace=np.empty(N+1)
		SOF_Trace[0]=self.Starting_SOF
		SOF=self.Starting_SOF
		FCRate=np.ones(len(self.trip_distances))*self.consumption_mixed
		FCRate[self.trip_mean_speeds<self.speed_thresholds[0]]=self.consumption_city
		FCRate[self.trip_mean_speeds>=self.speed_thresholds[1]]=self.consumption_highway
		Trip_SOF_Deltas=self.trip_distances*FCRate/self.Fuel_Tank_Capacity 
		#Main loop (forwards iteration)
		for k in np.arange(0,N,1):
			#Discharging
			SOF-=Trip_SOF_Deltas[k]
			#Charging - the optimal, admissable controls are selected for the current step based on current SOF
			admissable=Optimal_Control[k]>=0
			Optimal_U[k]=(np.around(np.interp(SOF,self.x[admissable],Optimal_Control[k][admissable]))*
				(self.SOF_Max-SOF)*self.Fuel_Tank_Capacity/self.Fueling_Rate)
			# print((Optimal_U[k]>0)*self.SOF_Max)
			SOF+=(Optimal_U[k]>0)*(self.SOF_Max-SOF)
			SOF_Trace[k+1]=SOF

		Dedicated_Energizing_Time=(np.zeros(N)+
			(Optimal_U>sys.float_info.epsilon)*(self.Fueling_Time_Penalty)+
			Optimal_U*(Optimal_U>sys.float_info.epsilon))

		# SIC=(Dedicated_Energizing_Time.sum()/60)/N
		SICD=(Dedicated_Energizing_Time.sum()/60)/(self.trip_distances.sum()/1000)


		return Optimal_U,SOF_Trace,SICD