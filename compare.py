""" Thesis - Comparing Model Distributions """

"""
This script is for doing a rank sums comparison of the Brier score results from test.py.
"""

### Imports ###
import pandas as pd
import csv
from scipy.stats import ranksums
import statistics

### Main ###
if __name__ == '__main__':
    # Reading the results file
    data = pd.read_csv('late_reintub_results_full.csv')
    
    # Putting results in proper format
    data = data.set_index('Model').T.to_dict('list')
    
    # Empty array for row data
    row = []
    
    # Opening csv to store p-values in
    with open('p_vals_late.csv','w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            # Writing the title row
            writer.writerow(['Model','null','Log Reg','RF','GBC'])
            # Looping through the results data
            for x in data:
                print(x + " " + str(statistics.mean(data[x])))
                # Emptying the row data array
                row = []
                # Putting model name in row
                row.append(str(x))
                # Rank sums test against the null
                if '_null' not in x and '_c' not in x and 'acc' not in x:
                    n = x + '_null'
                    row.append(ranksums(data[x], data[n])[1])
                    print(ranksums(data[x], data[n]))
                # Rank sums test against all other models
                for y in data:
                    if ('_null' not in x) and ('_null' not in y) and ('_c' not in x) and ('_c' not in y) and ('_acc' not in x) and ('_acc' not in y):
                        if x == y:
                            row.append('x')
                        else:
                            row.append(ranksums(data[x], data[y])[1])
                            print(ranksums(data[x], data[y]))
                # Ensuring only properly computed rows are included
                if len(row) > 1:
                    writer.writerow(row)

