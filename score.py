""" Thesis - Logistic Regression Scoring Systems """

"""
This script runs the logistic regression models on the optimal number
of features determined from the plots of performance vs. number of features
that are created in reduce.py
"""

import model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def fit_log_reg(X, y, significant):
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
    
    mod = model.log_reg_cv()
    
    mod.fit(X_train, y_train)
    pred = mod.predict_proba(X_test)[:,1]
    pred_acc = mod.predict(X_test)
    print('Max Predicted Probability:', end=' ')
    print(max(pred))
    print('Scores:')
    print(brier_score_loss(y_test, pred))
    print(roc_auc_score(y_test, pred))
    tn, fp, fn, tp = confusion_matrix(y_test, pred_acc).ravel()
    print('Confusion Matrix: ', end='')
    print(tn, fp, fn, tp)
    
    coef = list(mod.best_estimator_.coef_[0])
    colnames = list(colnames)
    for i in range(len(coef)):
        ind = coef.index(max(coef, key=abs))
        print(colnames[ind], coef[ind])
        colnames.pop(ind)
        coef.pop(ind)
        

significant_reintub = ['reintub_cpt_risk','pnapatos','asa_1_no_disturb','sepshockpatos','asa_5_moribund','asa_4_life_threat','ossipatos']

significant_early = ['early_reintub_cpt_risk','asa_1_no_disturb','sepshockpatos','pnapatos','asa_4_life_threat','surgspec_thoracic','asa_2_mild_disturb','ascites']

significant_late = ['late_reintub_cpt_risk','asa_1_no_disturb','sepshockpatos','pnapatos','asa_5_moribund','asa_4_life_threat','surgspec_cardiac_surgery']

# Getting set up
response = 'reintub'
surg, postop_complications, X, cpt_risk = model.set_up(response, True)

# Setting the y data
print(response)
y = surg.loc[:,response].values

fit_log_reg(X, y, significant_reintub)

print()

# Getting set up
response = 'early_reintub'
surg, postop_complications, X, cpt_risk = model.set_up(response, True)

# Setting the y data
print(response)
y = surg.loc[:,response].values

fit_log_reg(X, y, significant_early)

print()

# Getting set up
response = 'late_reintub'
surg, postop_complications, X, cpt_risk = model.set_up(response, True)

# Setting the y data
print(response)
y = surg.loc[:,response].values

fit_log_reg(X, y, significant_late)


