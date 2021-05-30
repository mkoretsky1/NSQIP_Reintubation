""" Thesis - Performance vs. Number of Features """

"""
This script creates plots of performance vs. number of features
based on the output from heuristic.py
"""

import model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def reduce_features(X, y, significant, model_name):
    drop = []
    # Dropping non-significant variables
    for column in X:
        if column != 'cpt' and column not in significant:
            drop.append(column)
    X = X.drop(drop, axis=1)
    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Map, impute, standardize
    X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
    
    # Getting the correct CV model
    if model_name == 'log_reg':
        mod = model.log_reg_cv()
    elif model_name == 'rf':
        mod = model.random_forest_cv()
    else:
        mod = model.gradient_boosting_cv()
    # Fiting the model, predicting and getting some scores
    mod.fit(X_train, y_train)
    pred = mod.predict_proba(X_test)[:,1]
    print('Max Predicted Probability:', end=' ')
    print(max(pred))
    print('Scores' + model_name + ':')
    print(brier_score_loss(y_test, pred))
    print(roc_auc_score(y_test, pred))
    # Setting the results from cv to the new model
    params = mod.best_params_
    if model_name == 'log_reg':
        mod = model.log_reg(params['C'], params['max_iter'])
    elif model_name == 'rf':
        mod = model.random_forest(params['n_estimators'], params['max_depth'], params['max_features'])
    else:
        mod = model.gradient_boosting(params['n_estimators'], params['max_depth'], params['max_features'])
    brier_scores = []
    c_stats = []
    # Looping
    for i in range(20):
        print(significant[-i-1])
        if i != 19:
            X = X.drop([significant[-i-1]], axis=1)
        print(X.shape)
        # Spltting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Map, impute, standardize
        X_train, X_test, colnames = model.map_impute_standardize(response, X_train, X_test, cpt_risk)
        # Fitting
        mod.fit(X_train, y_train)
        # Predicting
        pred = mod.predict_proba(X_test)[:,1]
        print('Max predicted probability:', end=' ')
        print(max(pred))
        pred_acc = mod.predict(X_test)
        print('Confustion matrix '+str(i)+':')
        tn, fp, fn, tp = confusion_matrix(y_test, pred_acc).ravel()
        print(tn, fp, fn, tp)
        # Appending
        brier_scores.append(brier_score_loss(y_test, pred))
        c_stats.append(roc_auc_score(y_test, pred))
    return brier_scores, c_stats
    
    

if __name__ == '__main__':
    
    # Variables
    significant_reintub_log_reg = ['pnapatos','asa_1_no_disturb','sepshockpatos','asa_5_moribund','asa_4_life_threat',
                                   'ossipatos','ascites','surgspec_gynecology','transt_outside_emergency_department','surgspec_thoracic',
                                   'surgspec_cardiac_surgery','fnstatus_totally_dependent','transt_from_acute_care_hospital_inpatient','asa_3_severe_disturb','wnd_2_clean/contaminated',
                                   'hxcopd','smoke','age_cat_75_85','asa_2_mild_disturb','prhct']
    
    significant_reintub_rf = ['asa_2_mild_disturb','asa_4_life_threat','prhct','electsurg','prbun',
                              'asa_1_no_disturb','sepsis_none','age_cat_65','prwbc','prplate',
                              'prcreat','hypermed','wnd_1_clean','hxcopd','asa_3_severe_disturb',
                              'dyspnea_no','fnstatus_independent','prsodm','surgspec_orthopedics','sepshockpatos']
    
    significant_reintub_gbc = ['asa_2_mild_disturb','prhct','age_cat_65','asa_4_life_threat','electsurg',
                               'prbun','prcreat','hypermed','asa_3_severe_disturb','asa_1_no_disturb',
                               'sepsis_none','prsodm','sepshockpatos','pnapatos','hxcopd',
                               'prplate','smoke','prwbc','wnd_1_clean','fnstatus_independent']
    
    significant_early_log_reg = ['asa_1_no_disturb','sepshockpatos','pnapatos','asa_4_life_threat','surgspec_thoracic',
                                 'asa_2_mild_disturb','ascites','surgspec_gynecology','asa_5_moribund','hxcopd',
                                 'smoke','electsurg','age_cat_65','hypermed','surgspec_orthopedics',
                                 'wnd_2_clean/contaminated','prcreat','sepsis_sepsis','prhct','age_cat_85']
    
    significant_early_rf = ['asa_2_mild_disturb','asa_1_no_disturb','prcreat','prwbc','prhct',
                            'prbun','prplate','electsurg','asa_4_life_threat','hxcopd',
                            'age_cat_65','sepsis_none','hypermed','asa_3_severe_disturb','dyspnea_no',
                            'sepshockpatos','surgspec_vascular','surgspec_orthopedics','smoke','surgspec_general_surgery']
    
    significant_early_gbc = ['asa_2_mild_disturb','prcreat','asa_4_life_threat','prhct','hypermed',
                             'age_cat_65','electsurg','hxcopd','prbun','asa_1_no_disturb',
                             'prplate','prwbc','asa_3_severe_disturb','sepshockpatos','sepsis_none',
                             'dyspnea_no','pnapatos','smoke','fnstatus_independent','surgspec_orthopedics']
    
    significant_late_log_reg = ['asa_1_no_disturb','sepshockpatos','pnapatos','asa_5_moribund','asa_4_life_threat',
                                'surgspec_cardiac_surgery','ossipatos','surgspec_gynecology','surgspec_thoracic','surgspec_neurosurgery',
                                'ascites','wnd_2_clean/contaminated','asa_3_severe_disturb','age_cat_75_85','prhct',
                                'diabetes_insulin','hxcopd','smoke','diabetes_no','sex']
    
    significant_late_rf = ['asa_2_mild_disturb','electsurg','prbun','age_cat_65','asa_1_no_disturb',
                           'asa_4_life_threat','prcreat','asa_3_severe_disturb','sepsis_none','wnd_1_clean',
                           'prplate','hxcopd','prsodm','late_reintub_cpt_risk','hypermed',
                           'sepshockpatos','dyspnea_no','fnstatus_independent','surgspec_orthopedics','prwbc']
    
    significant_late_gbc = ['asa_2_mild_disturb','prhct','age_cat_65','electsurg','prbun',
                            'asa_4_life_threat','prcreat','asa_1_no_disturb','hypermed','sepsis_none',
                            'hxcopd','wnd_1_clean','prplate','prsodm','prwbc','sepshockpatos',
                            'pnapatos','smoke','fnstatus_independent','sex','dyspnea_no']
    
    # Getting set up
    response = 'reintub'
    surg, postop_complications, X, cpt_risk = model.set_up(response, False)
    
    # Setting the y data
    print(response)
    y = surg.loc[:,response].values
    
    reintub_brier_scores, reintub_c_stats = reduce_features(X, y, significant_reintub_log_reg, 'log_reg')
    
    # Getting set up
    response = 'early_reintub'
    surg, postop_complications, X, cpt_risk = model.set_up(response, False)
    
    # Setting the y data
    print(response)
    y = surg.loc[:,response].values
    
    early_brier_scores, early_c_stats = reduce_features(X, y, significant_early_log_reg, 'log_reg')
    
    # Getting set up
    response = 'late_reintub'
    surg, postop_complications, X, cpt_risk = model.set_up(response, False)
    
    # Setting the y data
    print(response)
    y = surg.loc[:,response].values
    
    late_brier_scores, late_c_stats = reduce_features(X, y, significant_late_log_reg, 'log_reg')
        
    # Plotting performance vs.number of features for brier score
    num_predictors = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    plt.plot(num_predictors, reintub_brier_scores, label='combined')
    plt.plot(num_predictors, early_brier_scores, label='early')
    plt.plot(num_predictors, late_brier_scores, label='late')
    plt.legend(loc = 'upper right')
    plt.xlabel("Num Predictors")
    plt.xticks([0,5,10,15,20])
    plt.ylabel("Brier Score")
    plt.title("Gradient Boosting")
    plt.show()
    plt.clf()
    
    # Plotting performance vs. number of features for c-stat
    plt.plot(num_predictors, reintub_c_stats, label='combined')
    plt.plot(num_predictors, early_c_stats, label='early')
    plt.plot(num_predictors, late_c_stats, label='late')
    plt.legend(loc = 'lower right')
    plt.xlabel("Num Predictors")
    plt.xticks([0,5,10,15,20])
    plt.ylabel("C-Stat")
    plt.title("Gradient Boosting")
    plt.show()
    plt.clf()
    