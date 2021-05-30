""" Thesis - Testing of Models """

"""
This script is to test different models on 5 train/test splits for eventual comparison.
"""

### Imports ###
import numpy as np
import pandas as pd
import model
import random
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from scipy.stats import ranksums, sem
from statistics import mean
import warnings

### Main ###
if __name__ == '__main__':
    # Supressing warnings
    warnings.filterwarnings('ignore')
    
    # Specifying variable being analyzed
    response = 'reintub'
    
    # Getting set up
    surg, postop_complications, X, cpt_risk = model.set_up(response, False)
    
    # Instances left
    print('(Cases, Predictors): ' + str(X.shape))
    
    # Options for pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # Setting the y data
    y = surg.loc[:,response].values
    
    # Count in the category
    print('Instances of ' + response + ': ' + str(sum(y==1)))
    print()
    
    # Setting up models for comparison
    # Logistics regression
    log_reg = model.log_reg_cv()
    log_reg_scores = {'scores':[], 'null_scores':[], 'c_stat':[], 
                      'importances':[], 'p_mean':[], 'p_sd':[], 'acc':[]}
    
    # Random forest
    rf = model.random_forest_cv()
    rf_scores = {'scores':[], 'null_scores':[], 'c_stat':[],
                  'importances':[], 'p_mean':[], 'p_sd':[], 'acc':[]}
    
    # Gradient boosting
    gbc = model.gradient_boosting_cv()
    gbc_scores = {'scores':[], 'null_scores':[], 'c_stat':[], 
                  'importances':[], 'p_mean':[], 'p_sd':[], 'acc':[]}
    
    # Initial split for CV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Mapping cpt risk values, imputing and standardizing
    X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
    # Logistic regression training and testing
    model_name = 'log_reg_cv'
    log_reg, log_reg_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, log_reg, model_name, log_reg_scores)
    # Setting the results from cv to the new model
    params = log_reg.best_params_
    log_reg = model.log_reg(params['C'], params['max_iter'])
    
    # Random forest training and testing
    model_name = 'rf_cv'
    rf, rf_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, rf, model_name, rf_scores)
    params = rf.best_params_
    rf = model.random_forest(params['n_estimators'], params['max_depth'], params['max_features'])
    
    # Gradient boosting training and testing
    model_name = 'gbc_cv'
    gbc, gbc_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, gbc, model_name, gbc_scores)
    params = gbc.best_params_
    gbc = model.gradient_boosting(params['n_estimators'], params['max_depth'], params['max_features'])
    
    print('(tn, fp, fn, tp)')
    
    randoms = [42, 101, 666, 750]
    # Looping (4 non-cv model runs for each)
    for i in range(4):
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Mapping cpt risk values, imputing and standardizing
        X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
        
        # Logistic regression training and testing
        model_name = 'log_reg_' + str(i)
        log_reg, log_reg_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, log_reg, model_name, log_reg_scores)
        
        # Random forest training and testing
        model_name = 'rf_' + str(i)
        rf, rf_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, rf, model_name, rf_scores)
        
        # Gradient boosting training and testing
        model_name = 'gbc_' + str(i)
        gbc, gbc_scores = model.train_test_model(X_train, X_test, y_train, y_test, response, gbc, model_name, gbc_scores)
    
    # Specifying csv file name
    file_name = response + '_results.csv'
    # Writing the results to a csv file
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Model','1','2','3','4','5'])
        writer.writerow(['LogReg']+log_reg_scores['scores'])
        writer.writerow(['LogReg_null']+log_reg_scores['null_scores'])
        writer.writerow(['RF']+rf_scores['scores'])
        writer.writerow(['RF_null']+rf_scores['null_scores'])
        writer.writerow(['GBC']+gbc_scores['scores'])
        writer.writerow(['GBC_null']+gbc_scores['null_scores'])
        writer.writerow(['LogReg_c']+log_reg_scores['c_stat'])
        writer.writerow(['RF_c']+rf_scores['c_stat'])
        writer.writerow(['GBC_c']+gbc_scores['c_stat'])
        writer.writerow(['LogReg_acc']+log_reg_scores['acc'])
        writer.writerow(['RF_acc']+rf_scores['acc'])
        writer.writerow(['GBC_acc']+gbc_scores['acc'])
        
    # Getting average importances and standard errors for plotting
    log_reg_importances = np.array(log_reg_scores['importances'])
    rf_importances = np.array(rf_scores['importances'])
    gbc_importances = np.array(gbc_scores['importances'])
    print()
    # Comment out when running the only-cpt model
    model.plot_importances(colnames, log_reg_importances, 10, response, 'log_reg')
    model.plot_importances(colnames, rf_importances, 10, response, 'rf')
    model.plot_importances(colnames, gbc_importances, 10, response, 'gbc')
    
    # Getting average permuatation importances with standard deviations
    log_reg_means = np.array(log_reg_scores['p_mean'])
    log_reg_vars = np.square(np.array(log_reg_scores['p_sd']))
    rf_means = np.array(rf_scores['p_mean'])
    rf_vars = np.square(np.array(rf_scores['p_sd']))
    gbc_means = np.array(gbc_scores['p_mean'])
    gbc_vars = np.square(np.array(gbc_scores['p_sd']))
    # Averging the means and sds
    log_reg_means = log_reg_means.mean(axis=0)
    log_reg_sd = np.sqrt(log_reg_vars.mean(axis=0))
    rf_means = rf_means.mean(axis=0)
    rf_sd = np.sqrt(rf_vars.mean(axis=0))
    gbc_means = gbc_means.mean(axis=0)
    gbc_sd = np.sqrt(gbc_vars.mean(axis=0))
    
    # Looping to print significant values
    print('Logistic Regression Permuatation Importance')
    for i in log_reg_means.argsort()[::-1]:
        if log_reg_means[i] - 1.96 * log_reg_sd[i] > 0:
            print(colnames[i] + '\t' + str(log_reg_means[i]) + '\t+/-\t' + str(log_reg_sd[i]))
    print()
    print('Random Forest Permutation Importance')
    for i in rf_means.argsort()[::-1]:
        if rf_means[i] - 1.96 * rf_sd[i] > 0:
            print(colnames[i] + '\t' + str(rf_means[i]) + '\t+/-\t' + str(rf_sd[i])) 
    print()
    print('Gradient Boosting Permutation Importance')
    for i in gbc_means.argsort()[::-1]:
        if gbc_means[i] - 1.96 * gbc_sd[i] > 0:
            print(colnames[i] + '\t' + str(gbc_means[i]) + '\t+/-\t' + str(gbc_sd[i]))
        
    
    