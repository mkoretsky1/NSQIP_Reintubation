""" Thesis - Split Reintubation """

"""
This script is for splitting the full clean NSQIP 
file into reintubation and non-reintubation cases
"""

import pandas as pd
import model

# Reading in full datafile
surg = pd.read_csv('data/surg.csv', dtype={'cpt':str})
print(surg.shape)

# Separating early and late reintubation (early = 0-2 days postop)
surg.loc[:,'early_reintub'] = surg.loc[:,'dreintub'].apply(lambda x: 1 if 0 <= x <= 2 else 0)
surg.loc[:,'late_reintub'] = surg.loc[:,'dreintub'].apply(lambda x: 1 if x > 2 else 0)
print(surg.shape)

# Grouping by CPT for risk value calculation
cpt_risk = surg.groupby(surg.loc[:,'cpt'])
print('Number of CPT codes: ' + str(len(cpt_risk)))

# Generating CPT-specific risk values for each reintubation outcome
reintub_risk_dict = model.get_cpt_risks(surg, 'reintub', cpt_risk)
early_risk_dict = model.get_cpt_risks(surg, 'early_reintub', cpt_risk)
late_risk_dict = model.get_cpt_risks(surg, 'late_reintub', cpt_risk)
print(surg.shape)

# Isolating all reintub cases
reintub = surg[surg['reintub'] == 1]
print(reintub.shape)

# Isolating all non-reintub cases
n_reintub = surg[surg['reintub'] == 0]
print(n_reintub.shape)

# Sampling an equal number of non-reintub cases
n_reintub_reduced = n_reintub.sample(len(reintub))
print(n_reintub_reduced.shape)

# Creating balanced reintub dataset
surg_resampled = pd.concat([reintub, n_reintub_reduced])
print(surg_resampled.shape)
surg_resampled.to_csv('data/surg_resampled_reintub.csv')

# Isolating all early reintub cases
early_reintub = surg[surg['early_reintub'] == 1]
print(early_reintub.shape)

# Isloating the same amount of non-early reintub cases
n_early_reintub_reduced = n_reintub.sample(len(early_reintub))

# Creating balances early reintub dataset
surg_resampled = pd.concat([early_reintub, n_early_reintub_reduced])
surg_resampled.to_csv('data/surg_resampled_early.csv')

# Isolating all late reintub cases
late_reintub = surg[surg['late_reintub'] == 1]
print(late_reintub.shape)

# Isloating the same amount of non-late reintub cases
n_late_reintub_reduced = n_reintub.sample(len(late_reintub))

# Creating balanced early reintub dataset
surg_resampled = pd.concat([late_reintub, n_late_reintub_reduced])
surg_resampled.to_csv('data/surg_resampled_late.csv')