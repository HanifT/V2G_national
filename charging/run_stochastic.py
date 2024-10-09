import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import pickle as pkl
from datetime import datetime

import src
from src.reload import deep_reload

itineraries=pd.read_pickle('Data/Generated_Data/itineraries_selected.pkl')

itinerary_kwargs={'instances':5,
				  'tiles':5,
				  'home_charger_likelihood':1,
				  'work_charger_likelihood':0,
				  'destination_charger_likelihood':.1}

experiment_keys=[
	'home_charger_likelihood',
	'work_charger_likelihood',
	'destination_charger_likelihood',
]

experiment_levels=[
	[0,.125,.25,.375,.5],
	[0,.125,.25,.375,.5],
	[0,.125,.25,.375,.5],
]

experiment_inputs=src.experiments.GenerateExperiment(
	itinerary_kwargs,experiment_keys,experiment_levels)

exp_itineraries=src.experiments.SelectItineraries(itineraries,100,seed=1000)



solver_kwargs={'_name':'cbc','executable':'src/cbc','threads':4}
# solver_kwargs={'_name':'gurobi','solver_io':"python"}
# solver_kwargs={'_name':'cplex_direct'}

experiment_results=src.experiments.RunExperiment(
	exp_itineraries,experiment_inputs,iter_time=1,solver_kwargs=solver_kwargs)

now=datetime.now()
handle='output_'+now.strftime("%m%d%Y_%H%M%S")+'.pkl'

pkl.dump(experiment_results,open('Data/Generated_Data/'+handle,'wb'))