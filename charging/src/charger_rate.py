import os
import json
import pandas as pd
# Set up paths

class ChargerDataProcessor:
    def __init__(self, config_path: str = None, data_dir: str = None):
        current_dir = os.getcwd()

        # Set up configuration file path.
        if config_path is None:
            config_path = os.path.join(current_dir, 'charging', 'src', 'codex.json')
        self.config_path = config_path

        # Load configuration from JSON.
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Set up data folder path.
        if data_dir is None:
            data_dir = os.path.join(current_dir, 'charging', 'Data', 'AFDC')
        self.data_dir = data_dir

        # Load all required dataframes into a dictionary.
        self.dataframes = self._load_dataframes()

    def _load_dataframes(self) -> dict:
        # Mapping from file type to a function that reads the file.
        read_funcs = {
            'csv': lambda path, fi: pd.read_csv(path),
            'xlsx': lambda path, fi: pd.read_excel(path, sheet_name=fi.get('sheet', None))
        }

        dfs = {
            fi['name']: read_funcs[fi['type']](os.path.join(self.data_dir, fi['name']), fi)
            for fi in self.config['files']
            if os.path.exists(os.path.join(self.data_dir, fi['name']))
        }
        return dfs

    def compute_public_charger_rate(self) -> pd.DataFrame:
        # Access the DataFrames from the dictionary.
        afdc_df = self.dataframes.get('AFDC.csv')
        ev_register_df = self.dataframes.get('EV_register.xlsx')
        density_df = self.dataframes.get('density.xlsx')

        # Subset to the needed columns.
        afdc_df = afdc_df[["State", "EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]]

        # Compute station_counts as the number of rows (stations) per state.
        station_counts = afdc_df["State"].value_counts().rename_axis("State").to_frame("Total Stations")

        # Replace missing values with zero.
        afdc_df = afdc_df.fillna(0)

        # Sum charger counts per state (for Level 1, Level 2, and DC fast).
        state_charger_counts = afdc_df.groupby("State")[["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]].sum()

        # Total Chargers is the sum of the three charger columns for each state.
        state_charger_counts["Total Chargers"] = state_charger_counts.sum(axis=1)

        # Count the number of stations with DC fast chargers (where EV DC Fast Count > 0).
        state_charger_counts["Total_DC_stations"] = afdc_df.groupby("State")["EV DC Fast Count"].apply(lambda x: (x > 0).sum())

        # Count the number of stations with either a Level 1 or Level 2 charger ( > 0).
        state_charger_counts["Total_L_stations"] = afdc_df.groupby("State").apply(
            lambda df: ((df["EV Level1 EVSE Num"] > 0) | (df["EV Level2 EVSE Num"] > 0)).sum()
        )

        # Map state abbreviations to full state names.
        state_abbrev_to_name = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
            "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
            "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
            "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
            "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
            "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
            "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
            "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
            "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
            "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
            "WY": "Wyoming", "DC": "District of Columbia"
        }

        # Update indices with full state names.
        state_charger_counts.index = state_charger_counts.index.map(state_abbrev_to_name)
        state_charger_counts.index.name = 'State'

        station_counts.index = station_counts.index.map(state_abbrev_to_name)
        station_counts.index.name = 'State'
        station_counts = station_counts.reset_index()

        # Merge the state-level charger counts with station counts, EV registration, and density data.
        state_data = pd.merge(state_charger_counts, station_counts, left_on='State', right_on='State')
        state_data = pd.merge(state_data, ev_register_df, left_on='State', right_on='State')
        state_data = pd.merge(state_data, density_df[["State", "Population Density"]], left_on='State', right_on='State')

        # Use the median population density (D0) as a reference.
        D0 = state_data["Population Density"].median()

        # Compute various ratios.
        state_data["ratio_dc_fast"] = state_data["EV DC Fast Count"] / state_data["Registration Count"]
        state_data["ratio_dc_station"] = state_data["Total_DC_stations"] / state_data["Registration Count"]
        state_data["ratio_l"] = (state_data["EV Level1 EVSE Num"] + state_data["EV Level2 EVSE Num"]) / state_data["Registration Count"]
        state_data["ratio_l_station"] = state_data["Total_DC_stations"] / state_data["Registration Count"]
        state_data["charger_station_l_ratio"] = (state_data["EV Level1 EVSE Num"] + state_data["EV Level2 EVSE Num"]) / state_data["Registration Count"]
        state_data["charger_station_dc_ratio"] = (state_data["EV DC Fast Count"]) / state_data["Registration Count"]

        return state_data


