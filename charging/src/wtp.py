import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')

# %%
def calculate_wtp(income, bev_range):
    # Obtain the charging distance from the BEV range
    distance = bev_range * 0.6
    # Calculate WTP
    wtp = (0.29 * ((income / 198.85) ** 0.05398) * ((distance / 100) ** 1.49375)) / \
          (((income / 198.85) ** -0.03622) * ((distance / 100) ** 0.35168))

    return wtp