# A Machine Learning Approach to Predicting Early and Late Reintubation

Contains everything necessary to obtain the results from the Journal of Clinical Monitoring and Computing original research paper (Figure and Table numbers need to be updated).

doi: 10.1007/s10877-022-00908-z

A folder must be created that contains the raw ACS NSQIP data. 

1. clean.py:

      Takes the raw ACS NSQIP data and cleans it. Can change the name of the input/output files if you want to clean different subsamples.

2. split.py:

      Creates resampled data files for the combined, early and late reintubation outcomes so the number of events is equal to the number of non-events. This is also where the plots in Figure 1 come from.
      
3. model.py:

      Contains a bunch of different functions to help with modeling. Some code can be uncommented in the first function, called set_up, to run the models with the CPT-specific risk values as the only predictor. 
      
4. test.py

      For running the models on the full data. The outcome/response variable can be specified (i.e. reintub, early_reintub, late_reintub). Results are output to csv.
      
5. compare.py

      Takes the results csv from test.py and runs the rank sums test in terms of brier score. It also gets the average of each statistic and displays those (this is where the results from Tables 1-5 come from).
      
6. heuristic.py

      Performs the heruistic feature analysis for a specific outcome/response variable. 
      
7. reduce.py

      Takes the top 20 variables from the heuristic analysis (must be copied over manually at the moment) and creates plots of performance vs. number of features for both Brier score and c-statistic (this is where Figures 2-4 come from).
      
8. score.py

     For determining the coefficients from the logistic regression models fit using the optimal number of features from the plots of performance vs. number of features (this is where the results from Tables 6-9 come from).
