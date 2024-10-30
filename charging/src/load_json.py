import pandas as pd
import json
import os
# %%
# Path to the saved JSON file
data_dir = '/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017'
json_file_path = os.path.join(data_dir, 'dataframes.json')

# Load the JSON data into a dictionary of DataFrames
with open(json_file_path, 'r') as json_file:
    dataframes_json = json.load(json_file)

# Convert each JSON object back to a DataFrame
dataframes = {name: pd.DataFrame(data) for name, data in dataframes_json.items()}

# Example of accessing a specific DataFrame
for file_name, df in dataframes.items():
    print(f"\nData from {file_name}:")
    print(df.head())
