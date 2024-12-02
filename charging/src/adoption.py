import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import process
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry
# %%
# Adoption Rate
import glob
def combine_csv_files(file_path_pattern):
    # Match all files using the file path pattern
    file_path_pattern = "/Users/haniftayarani/V2G_national/charging/Data/Adoption/*.csv"
    csv_files = glob.glob(file_path_pattern)
    combined_data = pd.DataFrame()

    for file in csv_files:
        try:
            # Read the CSV file with no header
            df = pd.read_csv(file, header=None)  # Treat all rows as data, not headers

            # Extract the year
            year = int(df.loc[df[0] == 'Year', 1].values[0])

            # Perform necessary filtering on rows
            # Assuming the relevant data starts after a specific row index
            filtered_df = df.iloc[9:]  # Adjust index (e.g., `5`) based on your file structure
            filtered_df.columns = df.iloc[8]  # Set new header from row 4 (adjust as needed)
            filtered_df = filtered_df.reset_index(drop=True)  # Reset index

            # Ensure the relevant columns exist
            state_column = filtered_df.columns[0]  # Assuming the first column is the state name
            target_column = 'Privately-owned EVs on the road'

            # Extract and rename columns
            filtered_df = filtered_df[[state_column, target_column]].rename(columns={target_column: year})

            # Merge into the combined DataFrame
            if combined_data.empty:
                combined_data = filtered_df
            else:
                combined_data = pd.merge(combined_data, filtered_df, on=state_column, how="outer")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Ensure year columns are integers
    year_columns = [col for col in combined_data.columns if isinstance(col, int)]
    combined_data[year_columns] = combined_data[year_columns].astype(int)

    return combined_data