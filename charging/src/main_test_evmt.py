# %%
import warnings
warnings.filterwarnings("ignore")
import sys
import pickle
sys.path.append('D:\\Hanif\\V2G_national\\charging\\src')
from utilities import run_simulation
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("pyomo").setLevel(logging.WARNING)
# %% Input eVMT

# # Reading the input files
# codex_path = 'D:\\Hanif\\V2G_national\\charging\\codex.json'
# verbose = True
# data = process.load(codex_path)
# data_charging = data["charging"]
# data_trips = data["trips"]
# nstate_counts, state_abbrev_to_name = nhts_state_count()
#
# with open("D:\\Hanif\\V2G_national\\charging\\nhts_state_data.pkl", "wb") as f:
#     pickle.dump((nstate_counts, state_abbrev_to_name), f)

# %% execute optimal charging for eVMT
# results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=100, chunk=10, electricity_price_file="D:\\Hanif\\V2G_national\\charging\\src\\weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
# with open("results_dict_delay_100_10days.pkl", "wb") as f:
#     pickle.dump(results_dict_delay, f)
#
# with open("results_dict_delay_365_4days.pkl", "rb") as f:
#     delay_result_4 = pickle.load(f)

