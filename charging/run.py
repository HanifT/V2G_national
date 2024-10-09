import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import pickle as pkl
from datetime import datetime

import src
from src.reload import deep_reload

itineraries=pd.read_pickle('Data/Generated_Data/itineraries.pkl')

itinerary_kwargs={
	'tiles':5,
	'initial_sof':1,
	'final_sof':1,
    'battery_capacity':40*3.6e6,
	'home_charger_likelihood':1,
	'work_charger_likelihood':0,
	'destination_charger_likelihood':.1
}

experiment_keys=[
    'battery_capacity',
	'home_charger_likelihood',
	'work_charger_likelihood',
	'destination_charger_likelihood',
]

experiment_levels=[
    [10*3.6e6,35*3.6e6,60*3.6e6],
	[0,.25,.5],
	[0,.25,.5],
	[0,.25,.5],
]

experiment_inputs=src.experiments.GenerateExperiment(
	itinerary_kwargs,experiment_keys,experiment_levels)

exp_itineraries=src.experiments.SelectItineraries(itineraries,1000,seed=1000)

solver_kwargs={'_name':'cbc','executable':'src/cbc','threads':4}

vehicle_classes=[
    src.optimization.EVCSP,
    src.optimization.PHEVCSP,
    src.optimization.ICEVCSP,
]

experiment_results=src.experiments.RunExperiment(
    vehicle_classes,
	exp_itineraries,
    experiment_inputs,
    iter_time=1,
    solver_kwargs=solver_kwargs,
    disp=True,
    repititions=3,
)

now=datetime.now()
handle='output_'+now.strftime("%m%d%Y_%H%M%S")+'.pkl'

pkl.dump(experiment_results,open('Data/Generated_Data/'+handle,'wb'))