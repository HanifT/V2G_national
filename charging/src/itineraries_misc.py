# # Dictionary to map state codes to state names
# state_name_mapping = {
# 	'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
# 	'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
# 	'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
# 	'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
# 	'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan',
# 	'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
# 	'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
# 	'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
# 	'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
# 	'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
# 	'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
# 	'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
# }
#
# # Map state codes to full names
# unique_counts_by_state['state_name'] = unique_counts_by_state['HHSTATE'].map(state_name_mapping)
# # Assuming `dmv` is a DataFrame, not a function, access the columns like this
# unique_counts_by_state = pd.merge(unique_counts_by_state, dmv[['State', 'Total']], left_on="state_name", right_on="State", how="left")
# unique_counts_by_state['sampled_NHTS'] = unique_counts_by_state['Unique']
# unique_counts_by_state = unique_counts_by_state.drop(columns="Unique")
# unique_counts_by_state["Percent_sampled"] = (unique_counts_by_state["sampled_NHTS"]/unique_counts_by_state["Total"])*100
# unique_counts_by_state.to_csv("unique_counts_by_state.csv")
# Display the result
# print(unique_counts_by_state)