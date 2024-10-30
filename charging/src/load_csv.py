# %%
import pandas as pd
import glob
import os
# %%
# Set the directory path
data_dir = '/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017'

# Dictionary to store each DataFrame with its filename as the key
dataframes = {}

# Find all CSV files in the specified directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if csv_files:
    for file_path in csv_files:
        # Get the file name without the directory or extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Load the CSV file into a DataFrame and store it in the dictionary
        dataframes[file_name] = pd.read_csv(file_path)

        # Display the first few rows of each dataframe
        print(f"\nData from {file_name}:")
        print(dataframes[file_name].head())
else:
    print("No CSV files found in the directory.")

# %% Cleaning dataframes individually
# Example cleaning steps for each DataFrame (customize as needed)
for name, df in dataframes.items():
    # Clean the DataFrame: drop NA values, reset index
    cleaned_df = df.dropna().reset_index(drop=True)

    # Dynamically create individual DataFrame variables with their names
    globals()[name] = cleaned_df

    # Print to confirm that each DataFrame is now an individual variable
    print(f"\nCleaned data for {name}:")
    print(globals()[name].head())


trippub = trippub.drop([
    "TRPACCMP", "TRPHHACC", 'TRWAITTM', 'NUMTRANS', 'TRACCTM', 'DROP_PRK', 'TREGRTM', 'WHODROVE', 'TRPHHVEH',
    'HHMEMDRV', 'HH_ONTD', 'NONHHCNT', 'NUMONTRP', 'PSGR_FLG', 'DRVR_FLG', 'ONTD_P1', 'ONTD_P2', 'ONTD_P3',
    'ONTD_P4', 'ONTD_P5', 'ONTD_P6', 'ONTD_P7', 'ONTD_P8', 'ONTD_P9', 'ONTD_P10', 'ONTD_P11', 'ONTD_P12',
    'ONTD_P13', 'TRACC_WLK', 'TRACC_POV', 'TRACC_BUS', 'TRACC_CRL', 'TRACC_SUB', 'TRACC_OTH', 'TREGR_WLK',
    'TREGR_POV', 'TREGR_BUS', 'TREGR_CRL', 'TREGR_SUB', 'TREGR_OTH', 'DRVRCNT', 'NUMADLT', 'WRKCOUNT',
    'HHRESP', 'LIF_CYC', 'MSACAT', 'MSASIZE', 'RAIL', 'HH_RACE', 'HH_HISP', 'HH_CBSA', 'SMPLSRCE', 'R_AGE',
    'EDUC', 'R_SEX', 'PRMACT', 'PROXY', 'WORKER', 'DRIVER', 'WTTRDFIN', 'WHYTRP90', 'TRPMILAD', 'R_AGE_IMP',
    'R_SEX_IMP', 'OBHUR', 'DBHUR', 'OTHTNRNT', 'OTPPOPDN', 'OTRESDN', 'OTEEMPDN', 'OBHTNRNT', 'OBPPOPDN',
    'OBRESDN', 'DTHTNRNT', 'DTPPOPDN', 'DTRESDN', 'DTEEMPDN', 'DBHTNRNT', 'DBPPOPDN', 'DBRESDN'
], axis=1)

trip_veh = [1,2,3,4,5,6]
veh_type = [1,2,3,4,5]
trippub = trippub[trippub["TRPTRANS"].isin(trip_veh)]
trippub = trippub[trippub["TRPMILES"] > 0]
trippub = trippub[trippub["TRVLCMIN"] > 0]
trippub = trippub[trippub["VEHTYPE"].isin(veh_type)]
trippub = trippub[trippub["TRPMILES"] < 500]
# %% Saving all cleaned dataframes as a single JSON file
# Specify the file path to save the JSON file
json_file_path = '/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub_cleaned.json'

# Save the DataFrame as a JSON file
trippub.to_json(json_file_path, orient="records", lines=True)

print(f"DataFrame saved as JSON at {json_file_path}")

# %%
import matplotlib.pyplot as plt

# Define a function to plot the distribution of specified columns in a DataFrame
def plot_distributions(df, columns):
    for col in columns:
        if col in df.columns:
            # Apply specific capping based on column name
            capped_data = df[col].copy()
            if col == "TRPMILES":
                capped_data = capped_data.apply(lambda x: min(x, 100))  # Cap at 100 miles
            elif col == "TRVLCMIN":
                capped_data = capped_data.apply(lambda x: min(x, 200))  # Cap at 200 minutes

            # Plot the capped data
            plt.figure(figsize=(8, 5))
            plt.hist(capped_data.dropna(), bins=30, edgecolor='black')
            plt.title(f"Distribution of {col} (Capped)")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column '{col}' does not exist in the DataFrame.")

# Example usage
columns_to_plot = ["STRTTIME","ENDTIME", "TRPMILES", 'TRVLCMIN']  # Replace with your list of column names
plot_distributions(trippub, columns_to_plot)

