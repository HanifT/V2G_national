import pandas as pd
import requests
from census import Census


def get_charger_likelihood_by_state(year=2022):
    acs_api_key = "36bc9f2942111b56a8e1c8255e554dc53f79aca3"
    # Initialize Census API
    c = Census(acs_api_key)

    # Step 1: Get ACS5 Metadata
    def get_acs5_variables(year, dataset="acs/acs5"):
        url = f"https://api.census.gov/data/{year}/{dataset}/variables.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame.from_dict(data["variables"], orient="index").reset_index().rename(columns={"index": "name"})

    cached_acs5 = get_acs5_variables(year)

    # Step 2: Fetch ACS5 Data for Housing Units
    housing = pd.DataFrame(
        c.acs5.state(
            ["NAME"] + cached_acs5[cached_acs5["name"].str.startswith("B25024_")]["name"].tolist(),
            "*",  # Fetch all states
            year=year
        )
    )

    # Step 3: Reshape Data
    housing = housing.melt(id_vars=["NAME", "state"], var_name="variable", value_name="estimate")
    housing_vars = housing.merge(cached_acs5, left_on="variable", right_on="name", how="inner")

    # Step 4: Process 'units_in_structure'
    housing_vars[['est_moe', 'total', 'units_in_structure']] = housing_vars["label"].str.split("!!", expand=True).iloc[:, :3]
    housing_vars = housing_vars.drop(columns=["est_moe", "total"])
    housing_vars = housing_vars[housing_vars["estimate"].notna()]

    # Step 5: Define Charger Access Probability by Housing Type
    charger_probs = {
        "1, detached": 0.80,  # 80% chance of home charger
        "1, attached": 0.50,  # 60% chance of home charger
        "Multi-Unit Dwellings": 0.10  # 30% chance for apartments, etc.
    }

    # Step 6: Clean and Sort Data
    housing_vars = housing_vars[["NAME", "estimate", "units_in_structure"]]
    housing_vars = housing_vars.sort_values(by=["NAME", "estimate"], ascending=[True, False]).reset_index(drop=True)
    housing_vars["units_in_structure"] = housing_vars["units_in_structure"].fillna("Total housing units")

    # Step 7: Pivot Data for Computation
    housing_pivot = housing_vars.pivot(index="NAME", columns="units_in_structure", values="estimate")

    # Step 8: Compute Charger Availability
    total_households = housing_pivot["Total housing units"]
    detached_units = housing_pivot.get("1, detached", 0)
    attached_units = housing_pivot.get("1, attached", 0)
    mud_units = total_households - (detached_units + attached_units)  # Multi-Unit Dwellings

    housing_pivot["charger_likelihood"] = (
        (detached_units * charger_probs["1, detached"]) +
        (attached_units * charger_probs["1, attached"]) +
        (mud_units * charger_probs["Multi-Unit Dwellings"])
    ) / total_households

    # Step 9: Return Final Processed DataFrame
    return housing_pivot.reset_index()[["NAME", "charger_likelihood"]]


