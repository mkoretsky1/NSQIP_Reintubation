""" Thesis - Models """

"""
This script contains all functions needed for modeling.
"""

### Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_roc_curve, auc, confusion_matrix, accuracy_score, brier_score_loss, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import sem
import clean

### Functions ###
# set_up
def set_up(var, cpt):
    """
    Parameters
    ----------
    var : string
        variable being analyzed to pull up correct dataset and drop correct risk values.
    cpt : boolean
        wheter or not cpt-specific risk should be included
    Returns
    -------
    surg : pandas dataframe
        full surgial data read from csv.
    postop_complications : list
        list of postoperative complications for looping.
    X : pandas dataframe
        X data with proper variables dropped.
    cpt_risk : dictionary
        cpt groupings for risk value calculation.
    -------
    This function is for setting up for modeling in all other scripts.
    """
    # Reading proper file name based on the variable being analyzed (could be expanded for more outcomes)
    if var == 'reintub':
        file_name = 'data/surg_resampled_reintub.csv'
        risk_values_to_drop = ['early_reintub_cpt_risk','late_reintub_cpt_risk']
        current_risk_value = 'reintub_cpt_risk'
    elif var == 'early_reintub':
        file_name = 'data/surg_resampled_early.csv'
        risk_values_to_drop = ['reintub_cpt_risk','late_reintub_cpt_risk']
        current_risk_value = 'early_reintub_cpt_risk'
    else:
        file_name = 'data/surg_resampled_late.csv'
        risk_values_to_drop = ['reintub_cpt_risk','early_reintub_cpt_risk']
        current_risk_value = 'late_reintub_cpt_risk'
    surg = pd.read_csv(file_name, dtype={'cpt':str})
    # Sampling from the loaded file (for faster runs)
    # surg = surg.sample(10000)
    # Separating early and late reintubation (this is now done in split.py due to resampling)
    # surg.loc[:,'early_reintub'] = surg.loc[:,'dreintub'].apply(lambda x: 1 if 0 <= x <= 2 else 0)
    # surg.loc[:,'late_reintub'] = surg.loc[:,'dreintub'].apply(lambda x: 1 if x > 2 else 0)
    # Grouping by cpt for risk value calculation
    cpt_risk = surg.groupby(surg.loc[:,'cpt'])
    print('Number of CPT codes: ' + str(len(cpt_risk)))
    # Removing cases with lower CPT counts
    surg = remove_by_cpt(surg, cpt_risk)
    # Implementing specifications for reintub outcomes
    surg = reintub_specs(surg)
    # Looping to drop discharge variables
    for column in surg:
        if 'disch' in column:
            surg = surg.drop(column, axis=1)
    # Dropping bloodwork variables availiable for under 75% of cases
    surg = drop_bloodwork(surg)
    # Dropping PATOS variables (depends on specification of the analysis)
    # surg = drop_patos(surg)
    # List of postop complications to drop (also good for looping later)
    postop_complications = ['mortality','morbidity','oupneumo','reintub','cardiac','ssi',
                            'urninfec','othdvt','renal','othbleed','dreintub',
                            'early_reintub','late_reintub']
    # List of other variables to drop (no predictive value or previously categorized)
    other_drop = ['Unnamed: 0','caseid','admyr','age','weight','height','bmi','Unnamed: 0.1']
    # List of non-preoperative factors to drop
    non_preop_drop = ['optime','tothlos','doptodis','returnor','htooday']
    # Combining into one list and dropping to get the X data
    drop = postop_complications + other_drop + non_preop_drop + risk_values_to_drop
    # If the passed CPT parameter is false, remove the CPT-specific risk values for the current outcome
    if cpt ==  False:
        drop.append(current_risk_value)
    # Appending all other variables (for the CPT-only case, comment out otherwise)
    # for column in surg:
    #     if column != current_risk_value and column != 'cpt':
    #         drop.append(column)
    X = surg.drop(drop, axis=1)
    return surg, postop_complications, X, cpt_risk

# reintub_specs
def reintub_specs(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe without reintubation specifications implemented.
    Returns
    -------
    df : pandas dataframe.
        dataframe with reintubation specifications implemented.
    -------
    This function implements specifications that are specific to the reintubation outcomes.
    """
    # Removing cases where the patient is ventilator dependent
    print('Ventialtor dependent patients: ' + str(sum(df['ventilat']==1)))
    df = df[df['ventilat']==0]
    # Removing the ventilator dependence variable as well as the in/out variable
    df = df.drop(['ventilat', 'inout','ventpatos'], axis=1)
    # Removing cases where non-general anesthesia is used
    ## Note: Switch 1 to 0 to analyze non-general anesthesia cases
    print('Non-GA patients: ' + str(sum(df['anesthes_general']==0)))
    df = df[df['anesthes_general']==1]
    # Removing anesthesia variables iteratively
    for column in df:
        if 'anesthes' in column:
            df = df.drop(column, axis=1)
    # Note: above loop replaced by this for non-general analysis
    # df = df.drop('anesthes_general', axis=1)
    return df

# map_impute_standardize
def map_impute_standardize(response, X_train, X_test, cpt_risk):
    """
    Parameters
    ----------
    response : string
        complication being analyzed.
    X_train : pandas dataframe
        training data.
    X_test : pandas dataframe
        testing data.
    cpt_risk : dictionary
        dictionary with cpt:number_of_values pairs.
    Returns
    -------
    X_train : pandas dataframe
        training data.
    X_test : pandas dataframe
        testing data.
    colnames : list
        colnames for use in plotting.
    -------
    This function is for mapping cpt values, imputing, and standardizing the train and test data.
    """
    # Getting the risk value based on training set only (this is now done in split.py: minor data leakage)
    # risk_var = response + '_cpt_risk'
    # cpt_risk_dict = get_cpt_risks(X_train, response, cpt_risk)
    # Mapping these values to the test set
    # X_test.loc[:,risk_var] = X_test.loc[:,'cpt'].map(cpt_risk_dict)
    # Dropping cpt
    X_train = X_train.drop(['cpt'], axis=1)
    X_test = X_test.drop(['cpt'], axis=1)
    # Getting colnames for plotting
    colnames = X_train.columns
    # Imputing the data in each split respectively.
    X_train, X_test = impute_missing_values(X_train, X_test)
    # Rounding after imputation (need to convert back to df first)
    X_train = pd.DataFrame(X_train, columns=colnames)
    X_test = pd.DataFrame(X_test, columns=colnames)
    X_train = clean.round_df(X_train)
    X_test = clean.round_df(X_test)
    # Scaling the data in each split respectively
    X_train, X_test = standardize(X_train, X_test)
    return X_train, X_test, colnames

# impute_missing_values
def impute_missing_values(X_train, X_test):
    """
    Parameters
    ----------
    X_train : pandas dataframe
        non-imputed training data to be fit and transformed.
    X_test : pandas dataframe
        non-imputed testing data to be transformed.
    Returns
    -------
    X_train : pandas dataframe
        imputed training data.
    X_test : pandas dataframe
        imputed testing data.
    -------
    This function uses the iterative imputer to handle missing data.
    """
    imp = IterativeImputer()
    # Fitting and transforming training data
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    # Transforming testing data based on training fit (prevent data leakage)
    X_test = imp.transform(X_test)
    return X_train, X_test

# standardize
def standardize(X_train, X_test):
    """
    Parameters
    ----------
    X_train : pandas dataframe
        unstandardized values to be fit and transformed.
    X_test : pandas dataframe
        unstandardized values to be transformed.
    Returns
    -------
    X_train : pandas dataframe
        standardized values.
    X_test : pandas dataframe
        standarsized values based on X_train fit.
    -------
    This function is used to standardize the X data using standard scaler. The scaler
    is fit on the training data and applied to both the training and test data.
    """
    scaler = StandardScaler()
    # Fitting and transforming training data
    scaler.fit(X_train)
    scaler.transform(X_train)
    # Tranforming testing data based on traning fit (prevent data leakage)
    scaler.transform(X_test)
    return X_train, X_test

#get_cpt_risks
def get_cpt_risks(X, var, cpt_risk):
    """
    Parameters
    ----------
    X : pandas dataframe
        X values to generate cpt risks for.
    var : string
        variable name.
    cpt_risk : dictionary
        dictionary with cpt:number_of_values pairs.
    Returns
    -------
    X : pandas dataframe
        X values with cpt risks included.
    -------
    This function is for calculating the individual CPT-specific risk values for the 
    dataset. Calculated based off the training data and values mapped to the test data.
    """
    # Dictionary for risk values
    cpt_risk_dict = {}
    # Looping to get the average for each CPT
    for name, group in cpt_risk:
        cpt_risk_dict[name] = stats.mean(group.loc[:,var])
    # Create a new column using the dict for mapping
    risk_var = var + '_cpt_risk'
    X.loc[:,risk_var] = X.loc[:,'cpt'].map(cpt_risk_dict)
    # Plotting the CPT risks for each variable
    plot_cpt_risks(var, cpt_risk_dict)
    return cpt_risk_dict

# plot_cpt_risks
def plot_cpt_risks(var, cpt_risk_dict):
    """
    Parameters
    ----------
    var : string
        risk variable being plotted.
    cpt_risk_dict : dict
        dictionary with cpt:risk-value pairs.
    Returns
    -------
    None.
    -------
    This function is for plotting the CPT risk values for each outcome. 
    """
    # Plotting and labelling
    plt.hist(cpt_risk_dict.values(), bins=20)        
    plt.xlabel('CPT-specific risk value')
    plt.ylabel('Number of CPT codes')
    plt.title(var)
    # Saving the figure, displaying, and clearing
    file_name = 'figures/cpt/' + var + '_cpt_risks.png'
    plt.savefig(file_name)
    plt.show()
    plt.clf()
    
# remove_by_cpt
def remove_by_cpt(df, cpt_risk):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe to remove cases from based on number in each CPT code.
    cpt_risk : dictionary
        contains cpt risk information to base the removal on.
    -------
    Returns
    -------
    df : pandas dataframe
        dataframe with cases removed.
    -------
    This function is used to drop cases that have a CPT code with less than 25 cases.
    """
    # Getting a dictionary of how many surgeries belong to each CPT
    cpt_sizes_dict = dict(cpt_risk.size())
    # Mapping to the dataframe
    df.loc[:,'num_cpt'] = df.loc[:,'cpt'].map(cpt_sizes_dict)
    # Using only the most popular cpt
    # df = df[df['num_cpt']==max(df['num_cpt'])]
    # Using only cpt codes with more than 25 procedures
    df = df[df['num_cpt']>25]
    df = df.drop(['num_cpt'], axis=1)
    return df
    
# drop_patos
def drop_patos(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe with PATOS columns.
    Returns
    -------
    df : pandas dataframe
        dataframe without PATOS columns.
    ------
    This function is used to drop the PATOS variables from the analysis for 
    variable exploration.
    """
    for column in df:
        if 'patos' in column:
            df = df.drop(column, axis=1)
    return df

# drop_bloodwork
def drop_bloodwork(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe with bloodwork columns.
    Returns
    -------
    df : pandas dataframe
        dataframe without bloodwork columns.
    -------
    This function is used to drop preop bloodwork variables from the analysis when
    less than 75% of cases report the variable.
    """
    for column in df:
        if 'pr' in column:
            if (df[column].isnull().sum())/(len(df[column])) > 0.25: 
                df = df.drop(column, axis=1)
    return df    

# log_reg_cv
def log_reg_cv():
    """
    Returns
    -------
    log_reg : RandomizedSearchCV object
        CV model ready to be fit.
    -------
    Ths function sets up Logistic Regression with cross-validation
    """
    # Grid for cross-validation
    grid = {'C':[0.001,0.01,0.05,0.1,0.5,1.0], 'max_iter':[100,200,300,400,500]}
    # Defining the classifier and establishing CV
    log_reg = LogisticRegression(solver='liblinear', penalty='l2', random_state=None)
    log_reg_cv = RandomizedSearchCV(estimator=log_reg, param_distributions=grid, random_state=None,
                                    scoring='brier_score_loss', cv=5, n_iter=5)
    return log_reg_cv

def log_reg(c, max_iterations):
    """
    Parameters
    ----------
    c : float
        regulatization strength hyperparam from CV.
    max_iterations : int
        iterations hyperparam from CV.
    -------
    Returns
    -------
    log_reg : LogisticRegression object 
        logistic regression model ready to be fit.
    """
    log_reg = LogisticRegression(C=c, max_iter=max_iterations, solver='liblinear', 
                                 penalty='l2', random_state=None)
    return log_reg

# random_forest_cv
def random_forest_cv():
    """
    Returns
    -------
    forest_cv : RandomizedSearchCV object
        CV model ready to be fit.
    ------
    This function sets up Random Forest classifier with cross-validation.
    """
    # Grid for cross-validation
    grid = {'n_estimators':[10,25,50,75,100,150,200], 'max_depth':[2,3,4,5,6,7,8,9,10], 
            'max_features':[None,'sqrt','log2']}
    # Defining the classifier and establishing CV
    forest = RandomForestClassifier(criterion='entropy', random_state=None)
    forest_cv = RandomizedSearchCV(estimator=forest, param_distributions=grid, random_state=None,
                                   scoring='brier_score_loss', cv=5, n_iter=5)
    return forest_cv

def random_forest(n_estim, max_dep, max_feat):
    """
    Parameters
    ----------
    n_estim : int
        number of trees hyperparam from CV.
    max_dep : int
        depth of trees hyperparam from CV.
    max_feat : int
        features per tree hyperparam from CV.
    -------
    Returns
    -------
    forest : RandomForestClassifier obejct
        random forest model ready to be fit.
    """
    forest = RandomForestClassifier(criterion='entropy', n_estimators=n_estim, random_state=None,
                                    max_depth=max_dep, max_features=max_feat)
    return forest

# gradient_boosting_cv
def gradient_boosting_cv():
    """
    Returns
    -------
    gbc_cv : RandomizedSearchCV object
        CV model ready to be fit.
    -------
    This function sets up Gradient Boosting classifier with cross-validation and early stopping.
    """
    # Grid for cross-validation
    grid = {'n_estimators':[25,50,75,100,150,200], 'max_depth':[2,3,4,5,6,7,8,9,10],
            'max_features':[None,'sqrt','log2']}
    # Defining the classifier and establishing CV
    gbc = GradientBoostingClassifier(n_iter_no_change=20, random_state=None)
    gbc_cv = RandomizedSearchCV(estimator=gbc,param_distributions=grid, random_state=None,
                                scoring='brier_score_loss',cv=5,n_iter=5)
    return gbc_cv

def gradient_boosting(n_estim, max_dep, max_feat):
    """
    Parameters
    ----------
    n_estim : int
        number of trees hyperparam from CV.
    max_dep : int
        depth of trees hyperparam from CV.
    max_feat : int
        features per tree hyperparam from CV.
    -------
    Returns
    -------
    gbc : GradientBoostingClassifier object
        gradient boosting model ready to be fit.
    """
    gbc = GradientBoostingClassifier(n_iter_no_change=20, n_estimators=n_estim, random_state=None,
                                     max_depth=max_dep, max_features=max_feat)
    return gbc

# svc
def svc():
    """
    Returns
    -------
    svm_cv : RandomizedSearchCV object
        CV model ready to be fit.
    -------
    This function sets up Support Vector classifier with cross-validation.
    """
    # Grid for cross-validation 
    grid = {'C':[0.01,0.05,0.1,0.15,1.0], 'gamma':[1e-5,0.001,0.01,0.1,1,10]}
    # Defining classifier and bagging (maybe?)
    svm = SVC(kernel='sigmoid', probability=True, random_state=10)
    svm_cv = RandomizedSearchCV(estimator=svm,param_distributions=grid,random_state=10,
                                scoring='brier_score_loss',cv=5, n_iter=5)
    # bagging = BaggingClassifier(base_estimator=svm, random_state=10)
    return svm_cv

# knn
def knn():
    """
    Returns
    -------
    knn_cv : RandomizedSearchCVobject
        CV model ready to be fit.
    -------
    This function sets up KNN classifier with cross-validation.
    """
    # Grid for cross-validation
    grid = {'n_neighbors':[3,5,7,10,13,15,20,25]}
    # Defining classifier and establishing CV
    knn = KNeighborsClassifier()
    knn_cv = RandomizedSearchCV(estimator=knn,param_distributions=grid,scoring='brier_score_loss',
                                 cv=5,n_iter=5,random_state=10)
    return knn_cv

# train_test_model
def train_test_model(X_train, X_test, y_train, y_test, response, test_model, model_name, scores_dict, scoring_metric=None):
    """
    Parameters
    ----------
    X_train : pandas dataframe
        training data.
    X_test : pandas dataframe
        testing data.
    y_train : numpy array
        training labels.
    y_test : numpy array
        testing labels.
    response : string
        response variable.
    test_model : object
        model to be trained/tested.
    model_name : string
        name of model.
    scores_dict : dictionary
        dictionary containing lists with the model scores.
    scoring_metric : string
        metric to be used for calculation of permutation importances
    -------
    Returns
    -------
    scores_dict: dictionary
        dictionary containing lists with the model scores.
    -------
    This function is to train and test the models that are generated and looped over.
    """
    if scoring_metric == None:
        scoring_metric = 'accuracy'
    # Getting the overall rate based on the y training data
    overall_rate = (sum(y_train==1))/(len(X_train))
    # Fitting the models and dumping the fitted model into the pickel file
    test_model.fit(X_train, y_train)
    # If logistic regression, need the coefficients
    if 'log_reg' in model_name:
        pred = test_model.predict_proba(X_test)[:,1]
        pred_acc = test_model.predict(X_test)
        if type(test_model) == RandomizedSearchCV:
            predictors = test_model.best_estimator_.coef_[0]
        else:
            predictors = test_model.coef_[0]
    # Otherwise need the importances
    else:
        pred = test_model.predict_proba(X_test)[:,1]
        pred_acc = test_model.predict(X_test)
        if type(test_model) == RandomizedSearchCV:
            predictors = test_model.best_estimator_.feature_importances_
        else:
            predictors = test_model.feature_importances_
    tn, fp, fn, tp = confusion_matrix(y_test, pred_acc).ravel()
    print('Confusion Matrix: ', end='')
    print(tn, fp, fn, tp)
    # Getting accuracy scores and importances
    scores_dict['acc'].append(accuracy_score(y_test, pred_acc))
    scores_dict['importances'].append(predictors)
    # Getting permuatation importances and standard deviations
    perm = permutation_importance(test_model, X_test, y_test, n_repeats=10,
                                  scoring=scoring_metric)
    scores_dict['p_mean'].append(perm.importances_mean)
    scores_dict['p_sd'].append(perm.importances_std)
    # Getting the scores to return
    null_score = brier_score_loss(y_test, np.repeat(overall_rate,len(y_test)))
    scores_dict['null_scores'].append(null_score)
    score = brier_score_loss(y_test, pred)
    scores_dict['scores'].append(score)
    c_stat = roc_auc_score(y_test, pred)
    scores_dict['c_stat'].append(c_stat)
    return test_model, scores_dict

# plot_importances    
def plot_importances(names, importances, num, var, model):
    """
    Parameters
    ----------
    names : numpy array
        variable names.
    importances : numpy array
        importances/coefficients determined by model.
    num : int
        number of coefficients to plot.
    var : string
        variable for which predictors are being selected.
    model : string
        model being used as a string for graph output.
    Returns
    -------
    None.
    -------
    This function is for plotting the top predictors for each outcome when using LASSO regression
    """
    # Getting the average and standard error of the importances
    avg_importances = importances.mean(axis=0)
    std_devs = sem(importances, axis=0)
    # Initializing lists for the most important variable names and their importances
    plot_names = []
    plot_importances = []
    plot_sd = []
    # Converting to numpy arrays to lists (easier to work with)
    names = list(names)
    avg_importances = list(avg_importances)
    std_devs = list(std_devs)
    # For looking at plotted values
    print(model + " feature importances")
     # Looping based on input
    for i in range(num):
        # Getting the index of the max coeffieicent, storing name, value, and standard error
        ind = avg_importances.index(max(avg_importances, key=abs))
        plot_names.append(names[ind])
        plot_importances.append(avg_importances[ind])
        plot_sd.append(std_devs[ind]*1.96)
        # Taking a look at the values being plotted
        print(names[ind], avg_importances[ind], std_devs[ind])
        # Removing the max for the next iteration
        avg_importances.pop(ind)
        names.pop(ind)
        std_devs.pop(ind)
    print()
    # Plotting
    plt.barh(plot_names, plot_importances, xerr = plot_sd)
    plt.gca().invert_yaxis()
    plt.grid(True, which='major', axis='both')
    plt.ylabel('Predictors')
    plt.title(var + ': ' + model)
    plt.tight_layout()
    # Saving the figure to the proper folder based on the file name
    if 'log_reg' in model:
        plt.xlabel('OR Estimates')
        file_name = 'figures/log_reg/' + var + '_' + model + '_top_predictors.png'
    else:
        plt.xlabel('Importances')
        if 'rf' in model:
            file_name = 'figures/rf/' + var + '_' + model + '_top_predictors.png'
        else:
            file_name = 'figures/gbc/' + var + '_' + model + '_top_predictors.png'
    plt.savefig(file_name)
    # Displaying the figure and clearing
    plt.show()
    plt.clf()