""" Thesis - Heruistic """

"""
This script perfroms the heuristic feature analysis
"""

import model
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import warnings

if __name__ == '__main__':
    # Supressing warnings
    warnings.filterwarnings('ignore')
    heuristic_dict = {}
    
    # Looping
    for i in range(8):
        # Only including CPT-specific risk in half the runs
        cpt = True
        if i % 2 != 0:
            cpt = False
        
        # Setting the scoring metric based on the iteration
        scoring_metric = ''
        if i < 4:
            scoring_metric = 'brier_score_loss'
        else:
            scoring_metric = 'roc_auc'
    
        # Getting set up
        response = 'late_reintub'
        surg, postop_complications, X, cpt_risk = model.set_up(response, cpt)
        
        # Setting the y data
        print(response)
        y = surg.loc[:,response].values
        
        # Model name - can be chosen to be any of the three
        model_name = 'log_reg'
        mod = model.log_reg_cv()
        # model_name = 'rf'
        # mod = model.random_forest_cv()
        # model_name = 'gbc'
        # mod = model.gradient_boosting_cv()
    
        # Scores dictionary
        mod_scores = {'scores':[], 'null_scores':[], 'c_stat':[], 
                      'importances':[], 'p_mean':[], 'p_sd':[], 'acc':[]}
        
        print(model_name)
        
        print(scoring_metric)
        
        print()
        
        # Spltting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Map, impute, standardize
        X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
        
        # Cross validation training 
        mod, mod_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, mod, model_name, mod_scores, scoring_metric)
        # Setting the params (based on model name)
        params = mod.best_params_
        if model_name == 'log_reg':
            mod = model.log_reg(params['C'], params['max_iter'])
        elif model_name == 'rf':
            mod = model.random_forest(params['n_estimators'], params['max_depth'], params['max_features'])
        else:
            mod = model.gradient_boosting(params['n_estimators'], params['max_depth'], params['max_features'])
    
     
        # Looping (4 non-cv model runs for each)
        for i in range(4):
            # Splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # Mapping cpt risk values, imputing and standardizing
            X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
            # Gradient boosting training and testing
            model_name = model_name + str(i)
            mod, mod_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, mod, model_name, mod_scores, scoring_metric)
        
        # Getting the average importances, average permuatation importances
        importances = np.array(mod_scores['importances'])
        avg_importances = importances.mean(axis=0)
        avg_importances = list(avg_importances)
        means = np.array(mod_scores['p_mean'])
        variances = np.square(np.array(mod_scores['p_sd']))
        means = means.mean(axis=0)
        sd = np.sqrt(variances.mean(axis=0))
        
        # Making the colnames into a list
        colnames_list = list(colnames)
        
        print()
        
        # Increasing dictionary count for top 10 importances
        for i in range(10):
            ind = avg_importances.index(max(avg_importances, key=abs))
            print(colnames_list[ind])
            if colnames_list[ind] not in heuristic_dict:
                heuristic_dict[colnames_list[ind]] = [0,0]
            heuristic_dict[colnames_list[ind]][0] += 1
            avg_importances.pop(ind)
            colnames_list.pop(ind)
            
        print()
        
        # Increasing dictionary count for the significant permuatation importances
        for i in means.argsort()[::-1]:
            if means[i] - 1.96 * sd[i] > 0:
                print(colnames[i])
                if colnames[i] not in heuristic_dict:
                    heuristic_dict[colnames[i]] = [0,0]
                heuristic_dict[colnames[i]][1] += 1
                
        print()
    
    total_dict = {}
    
    # Sorting the heuristic dictionary by the total count
    for key in heuristic_dict:
        if key not in total_dict:
            total_dict[key] = 0
        total_dict[key] = sum(heuristic_dict[key])
    
    # Printig out the sorted total count dictionary
    print(sorted(total_dict.items(), key = lambda item: item[1], reverse=True))
        
        
        
    