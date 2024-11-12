# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from reload import deep_reload
from utilities import load_itineraries, read_pkl
import optimization
import experiments
import process
# %%

# Reading the input files
codex_path = '/Users/haniftayarani/V2G_national/charging/codex.json'
verbose = True
data = process.load(codex_path)
data_cahrging = data["charging"]
data_trips = data["trips"]

data_cahrging_summary, data_cahrging_grouped = process.plot_days_between_charges(data_cahrging, min_energy=10, min_days=-1, max_days=15, excluded_makes=["Toyota"])

# %%
# file_path = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
# file_path = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'
# file_path = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl'
# data = read_pkl(file_path)
# %%
#Importing itineraries
itineraries = load_itineraries(day_type="all")  # Change to False for weekend
# %%

solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc', "thread": 200}

itinerary_kwargs = {
    "tiles": 7,
    'initial_soc': 1,
    'final_soc': 1,
    'battery_capacity': 66*3.6e6,
    'home_charger_likelihood': 0.9,
    'work_charger_likelihood': 0.50,
    'destination_charger_likelihood': 0.1,
}
# Initialize an empty list to collect the results
all_tailed_itineraries = []

# Set the number of itineraries to loop through
num_itineraries = len(itineraries)
num_itineraries = 1000

# Solver and itinerary parameters
solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}

# Loop over each itinerary in the list
for n in range(num_itineraries):
    # Instantiate the EVCSP class for each itinerary
    problem = optimization.EVCSP(itineraries[n], itinerary_kwargs=itinerary_kwargs)
    problem.Solve(solver_kwargs)

    # Print status and SIC for tracking
    print(f'Itinerary {n}: solver status - {problem.solver_status}, termination condition - {problem.solver_termination_condition}')
    print(f'Itinerary {n}: SIC - {problem.sic}')

    # Repeat (tail) the itinerary across tiles
    tiles = itinerary_kwargs["tiles"]
    tailed_itinerary = pd.concat([itineraries[n]['trips']] * tiles, ignore_index=True)

    # Add the SOC from the solution to the tailed itinerary
    soc_values = problem.solution['soc', 0].values  # Extract SOC values from problem.solution
    tailed_itinerary['SOC'] = soc_values[:len(tailed_itinerary)]

    # Add the SIC as a column to the tailed itinerary (repeated for each row)
    tailed_itinerary['SIC'] = problem.sic

    # Append the tailed itinerary with SOC and SIC to the results list
    all_tailed_itineraries.append(tailed_itinerary)

# Concatenate all the individual DataFrames into one final DataFrame
final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
# %%

final_df = process.add_order_column(final_df)
final_df_days = process.assign_days_of_week(final_df)
final_df_days = process.identify_charging_sessions(final_df_days)
final_df_days, final_df_days_grouped = process.calculate_days_between_charges_synt(final_df_days)

# %%

line_kwargs = {
    'linewidth': 3,
}

axes_kwargs = {
    'facecolor': 'whitesmoke',
    'ylim': [0, 1],
    'xlabel': 'Trip/Park Event',
    'ylabel': 'SOC [-]',
}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
if not hasattr(ax, '__iter__'):
    ax = [ax]

ax[0].plot(problem.solution.soc)
# ax[0].plot(problem.solution.sof)
# ax[0].plot(problem.solution.u_cd_cs[0])

_ = [ax.grid(ls='--') for ax in ax]
plt.show()


# %%

itinerary_kwargs = {
                'tiles':5,
                'initial_sof':1,
                'final_sof':1,
                'battery_capacity':82*3.6e6,
                'home_charger_likelihood':1,
                'work_charger_likelihood':0.75,
                'destination_charger_likelihood':.25
}

experiment_keys=[
                'battery_capacity',
                'home_charger_likelihood',
                'work_charger_likelihood',
                'destination_charger_likelihood',
]

experiment_levels = [
                    [75*3.6e6, 66*3.6e6, 60*3.6e6],
                    [0.5, .75, 1],
                    [0.25, 0.5, 0.75],
                    [0, .25, .5],
]

experiment_inputs = experiments.GenerateExperiment(itinerary_kwargs, experiment_keys, experiment_levels)
# 162000
exp_itineraries = experiments.SelectItineraries(itineraries, 2, seed=1)


solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc', 'threads': 15}

vehicle_classes = [optimization.EVCSP]

experiment_results = experiments.RunExperiment(
                     vehicle_classes,
                     exp_itineraries,
                     experiment_inputs,
                     iter_time=1,
                     solver_kwargs=solver_kwargs,
                     disp=False,
                     repititions=3,
)

out = experiments.ProcessResults(experiment_results)
# %%
optimization.EVCSP.solution.soc
