import sys
import time
import argparse
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import pyomo.environ as pyomo

from copy import copy,deepcopy
from datetime import datetime

from .utilities import IsIterable,FullFact,ProgressBar,TimeLimit,CondPrint
from .optimization import RunDeterministic,RunStochastic,RunAsIndividuals

import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def ResultsDataFrame(results):

	n=max([len(val) for val in results[0].values() if IsIterable(val)]+[1])

	out={}

	for key in results[0].keys():

		out[key]=[]

	for idx,result in enumerate(results):
		for key,val in result.items():

			if IsIterable(val):
				out[key]+=[np.nanmean(val[val>=0])]

			else:
				out[key]+=[val]

	out_df=pd.DataFrame.from_dict(out)

	return out_df

def ExtendResultsDataFrame(results):

	results_1=results.copy()

	results['sic']=results['sic_expected']
	results['type']=['s']*results.shape[0]

	results_1['sic']=results_1['sic_mean']
	results_1['type']=['d']*results.shape[0]

	return pd.concat([results,results_1])

def SelectItineraries(itineraries,size=1,seed=None):

	return np.random.default_rng(seed).choice(itineraries,size=size,replace=False)

def GenerateExperiment(inputs,keys,levels):

	design=FullFact([len(level) for level in levels])

	experiment_inputs=[]

	for idx,row in enumerate(design):

		row_dict=inputs.copy()

		for idx1,key in enumerate(keys):

			row_dict[key]=levels[idx1][row[idx1]]

		experiment_inputs.append(row_dict)

	return experiment_inputs

def ProcessResults(results):

	n=len(results)

	keys=results[0].keys()

	out={}

	for key in keys:

		if hasattr(results[0][key],'__iter__'):
			# print(results[0][key].shape[1])

			for idx in range(results[0][key].shape[1]):

				out_array=np.zeros(n)

				for idx1,res in enumerate(results):

					out_array[idx1]=np.nanmean(res[key][:,idx])

				out[f'{key}_{idx}']=out_array
		else:

			out_array=np.zeros(n)

			for idx1,res in enumerate(results):

				out_array[idx1]=res[key]

			out[key]=out_array

	return pd.DataFrame.from_dict(out)



def RunExperiment(
	vehicle_classes,itineraries,inputs,iter_time=1,solver_kwargs={},disp=True,repititions=1):

	for idx0,input_dict in enumerate(inputs):

		CondPrint(f'Case {idx0+1} of {len(inputs)}\n',disp=disp)

		sic_mean=np.ones((len(itineraries),len(vehicle_classes)))*-1
		events=np.ones((len(itineraries),len(vehicle_classes)))*-1
		distance=np.ones((len(itineraries),len(vehicle_classes)))*-1

		for idx1 in ProgressBar(range(len(itineraries)),disp=disp):

			for idx2,vehicle_class in enumerate(vehicle_classes):

				sic_reps=[np.nan]*repititions
				events_reps=[np.nan]*repititions
				distance_reps=[np.nan]*repititions

				for idx3 in range(repititions):

					try:
						with TimeLimit(iter_time):

							sic_reps[idx3],problem=RunDeterministic(
								vehicle_class,
								itineraries[idx1],
								itinerary_kwargs=input_dict,
								solver_kwargs=solver_kwargs
								)

							events_reps[idx3]=problem.inputs['n_e']
							distance_reps[idx3]=problem.inputs['l_i']
							
					except:
						pass

				# print(events,np.nanmean(events))
				sic_mean[idx1,idx2]=np.nanmean(sic_reps)
				events[idx1,idx2]=np.nanmean(events_reps)
				distance[idx1,idx2]=np.nanmean(distance_reps)


		input_dict['sic_mean']=sic_mean
		input_dict['events']=events
		input_dict['distance']=distance

	return inputs

def RunExperimentStochastic(itineraries,inputs,iter_time=1,solver_kwargs={},disp=True):

	for idx0,input_dict in enumerate(inputs):

		CondPrint(f'Case {idx0+1} of {len(inputs)}\n')

		sic_expected=np.ones(len(itineraries))*-1
		sic_mean=np.ones(len(itineraries))*-1
		events=np.ones(len(itineraries))*-1
		distance=np.ones(len(itineraries))*-1

		for idx1 in ProgressBar(range(len(itineraries))):

			try:
				with TimeLimit(iter_time):

					sic_s,problem_s=RunStochastic(itineraries[idx1],
										itinerary_kwargs=input_dict,
										solver_kwargs=solver_kwargs)

					sic_d,_=RunAsIndividuals(problem_s,solver_kwargs=solver_kwargs)

					sic_expected[idx1]=np.nanmean(sic_s)
					sic_mean[idx1]=np.nanmean(sic_d)
					events[idx1]=problem_s.inputs['n_e']
					distance[idx1]=problem_s.inputs['l_i']

			except:
				pass

		input_dict['sic_expected']=sic_expected
		input_dict['sic_mean']=sic_mean
		input_dict['events']=events
		input_dict['distance']=distance

	return inputs