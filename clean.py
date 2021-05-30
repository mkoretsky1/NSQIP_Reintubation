""" Thesis - Data Cleaning and Encoding """

"""
This script takes the raw ACS NSQIP surgery data and produces  
a clean csv file that is ready for computation and analysis.
It handles data cleaning, encoding, and preprocessing.
"""

### Imports ###
import pandas as pd
import numpy as np

### Functions ###
# clean_df
def clean_df(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe to be cleaned.
    Returns
    -------
    df : pandas dataframe
        cleaned dataframe.
    -------
    This function cleans the passed dataframe by dropping unwanted variables,
    encoding null values, making changes to specific variables, and dropping
    unecessary RVU variables.
    """
    # Opening unwanted variables file and reading them into array
    ## Note: this needs to be switched to unwanted_vars_full when using the full data file
    f = open('data/unwanted_vars_full.txt')
    unwanted = []
    for line in f:
        # Getting rid of newline character
        line = line[:-1]
        unwanted.append(line)
    # Dropping unwanted variables
    df = df.drop(unwanted, axis=1)
    # -99, NULL, Unknown, and Unknown/Not Reported represent missing values, changing to nan
    df = df.replace(-99, np.nan)
    df = df.replace('-99', np.nan)
    df = df.replace('NULL', np.nan)
    df = df.replace('Unknown', np.nan)
    df = df.replace('Unknown/Not Reported', np.nan)
    # Making changes to some categorical variables to match user guide
    # Specifically for age (for creating age categories)
    df = df.replace('90+', 90)
    # Specifically for transfer status
    df = df.replace('Admitted directly from home','Not transferred (admitted from home)')
    df = df.replace('Acute Care Hospital','From acute care hospital inpatient')
    df = df.replace('Chronic Care Facility','Nursing home - Chronic care - Intermediate care')
    df = df.replace('VA Chronic Care Facility','Nursing home - Chronic care - Intermediate care')
    df = df.replace('VA Acute Care Hospital','From acute care hospital inpatient')
    df['transt'] = df['transt'].replace('Other','Transfer from other')
    # Specifically for anesthesia
    df = df.replace('MAC/IV Sedation','Monitored Anesthesia care (MAC) / IV Sedation')
    df = df.replace('Monitored Anesthesia Care','Monitored Anesthesia care (MAC) / IV Sedation')
    # Specifically for diabetes
    df = df.replace('ORAL','NON-INSULIN')
    # Specifically for othbleed
    df = df.replace('Bleeding/Transfusions','Transfusions/Intraop/Postop')
    # Changing age to float type for encoding
    df['age'] = df['age'].astype(float)
    # Dropping any remaining RVU columns as they are unnecessary
    for column in df:
        if 'rvu' in column:
            df = df.drop(column, axis=1)
    return df

# encode_vars
def encode_vars(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe to be encoded.
    Returns
    -------
    df : pandas dataframe
        encoded dataframe.
    -------
    This function encodes some of the variables in the dataframe by creating
    age categories, encoding binary variables, dealing with specific-case binary variables,
    encoding postop complication variables, dropping number of occurrances variables,
    creating dummy variables, replacing spaces, and renaming some fringe cases.
    """
    # Creating caegories for age using a helper function
    df['age_cat'] = df['age'].apply(age_categories)
    # List of postop variables
    postop = ['supinfec','wndinfd','orgspcssi','dehis','oupneumo','reintub',
                  'pulembol','failwean','renainsf','oprenafl','urninfec','cnscva','cdarrest',
                  'cdmi','othbleed','othdvt','othsysep','othseshock']
    # Encoding non-postop variables into 1/0
    for var in df:
        # Changing Yes/No to 1/0
        if df[var].nunique() == 2 and (df[var].dtype == object or df[var].dtype == str) and var != 'sex' and var != 'inout' and var not in postop:
            df[var] = df[var].map({'Yes':1,'No':0})
        elif df[var].nunique() == 1 and (df[var].dtype == object or df[var].dtype == str) and var != 'sex' and var != 'inout' and var not in postop:
            df[var] = df[var].map({'Yes':1,'No':0})
        # Dealing with other binary variables - gender and in/out status
        elif var == 'sex':
            df[var] = df[var].map({'male':1,'female':0})
        elif var == 'inout':
            df[var] = df[var].map({'Outpatient':1,'Inpatient':0})
    # Encoding postop complications into 1/0
    for var in postop:
        # Encoding the variables using a helper function
        df[var] = df[var].apply(encode_postop)
        # Dropping the number of occurrances variables (not relavent)
        num_var = 'n' + var
        df = df.drop(num_var, axis=1)
    # Creating dummy variables (one-hot encoding)
    df = pd.get_dummies(df,columns=['race_new','transt','dischdest','anesthes','surgspec','diabetes',
                                    'dyspnea','fnstatus2','prsepis','wndclas','asaclas','age_cat']
                        ,prefix=['race','transt','dischdest','anesthes','surgspec','diabetes',
                                 'dyspnea','fnstatus','sepsis','wnd','asa','age_cat'])
    # Replacing spaces/underscores in dummy variables
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = [c.replace('_-_','_') for c in df.columns]
    df.columns = [c.replace('-','_') for c in df.columns]
    # Editing some fringe cases to match user guide
    df = df.rename(columns={'transt_not_transferred_(admitted_from_home)':'transt_not_transferred',
                       'dischdest_against_medical_advice_(ama)':'dischdest_against_medical_advice',
                       'anesthes_monitored_anesthesia_care_(mac)_/_iv_sedation':'anesthes_mac',
                       'surgspec_otolaryngology_(ent)':'surgspec_otolaryngology_ent',
                       'age_cat_85+':'age_cat_85','age_cat_<65':'age_cat_65',
                       'dischdest_skilled_care,_not_home':'dischdest_skilled_care_not_home'})
    return df

# age_categories
def age_categories(age):
    """
    Parameters
    ----------
    age : float
        age as an float.
    Returns
    -------
    str
        categorization of age.
    -------
    This helper function is used to create age categories that match the user guide.
    """
    if age == 'nan':
        return np.nan
    else:
        #age = age.astype(int)
        if age < 65:
            return '<65'
        elif 65 <= age < 75:
            return '65_75'
        elif 75 <= age < 85:
            return '75_85'
        elif age >= 85:
            return '85+'

# encode_postop
def encode_postop(var):
    """
    Parameters
    ----------
    var : postop variable (string)
        postop description.
    Returns
    -------
    int
        encoded postop description.
    -------
    This helper function is used to encode the postop complication variables.
    """
    if var == 'No Complication':
        return 0
    elif var == np.nan:
        return np.nan
    else:
        return 1    

# aggregate response variables
def aggregate_response(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe to be aggregated.
    Returns
    -------
    df : pandas dataframe
        aggregated dataframe.
    -------
    This function is for aggregating the postoperative complication response variables
    that are made up of multiple complications or need to be converted from a numeric value.
    """
    # Aggregating mortality from numeric "days from operation until death" variable
    df['mortality'] = df['dopertod'].apply(lambda x: 1 if x >= 0 else 0)
    # Note: all of the following employ helper functions
    # Aggregating cardiac event (cardiac arrest or myocardial infarction)
    df['cardiac'] = df.apply(aggregate_cardiac, axis=1)
    # Aggregating renal failure
    df['renal'] = df.apply(aggregate_renal, axis=1)
    # Aggregating SSI
    df['ssi'] = df.apply(aggregate_ssi, axis=1)
    # Aggregating morbidity
    df['morbidity'] = df.apply(aggregate_morbidity, axis=1)
    # Dropping the variables used for aggregation
    df = df.drop(['cdarrest','cdmi','renainsf','oprenafl','supinfec','orgspcssi','wndinfd',
                  'dehis','pulembol','failwean','cnscva','othsysep','othseshock','dopertod'], 
                 axis=1)
    return df

# aggregate_cardiac
def aggregate_cardiac(row):
    """
    Parameters
    ----------
    row : pandas dataframe row
        row to be aggregarted for cardiac outcome.
    Returns
    -------
    int
        encoded cardiac outcome aggregation.
    -------
    This is a helper function for cardiac variable aggregation.
    """
    if row.cdarrest == 1 or row.cdmi == 1:
        return 1
    else:
        return 0

# aggregate_renal
def aggregate_renal(row):
    """
    Parameters
    ----------
    row : pandas dataframe row
        row to be aggregarted for renal outcome.
    Returns
    -------
    int
        encoded renal outcome aggregation.
    -------
    This is a helper function for renal failure variable aggregation.
    """
    if row.renainsf == 1 or row.oprenafl == 1:
        return 1
    else:
        return 0
    
# aggregate_ssi
def aggregate_ssi(row):
    """
    Parameters
    ----------
    row : pandas dataframe row
        row to be aggregarted for ssi outcome.
    Returns
    -------
    int
        encoded ssi outcome aggregation.
    -------
    This is a helper function for ssi variable aggregation.
    """
    if row.supinfec == 1 or row.orgspcssi == 1 or row.wndinfd == 1:
        return 1
    else:
        return 0
    
# aggregate_morbidity
def aggregate_morbidity(row):
    """
    Parameters
    ----------
    row : pandas dataframe row
        row to be aggregarted for morbidity outcome.
    Returns
    -------
    int
        encoded morbidity outcome aggregation.
    -------
    This is a helper function for morbidity variable aggregation.
    """
    morb = ['cdarrest','cdmi','renainsf','oprenafl','supinfec','orgspcssi','wndinfd',
            'dehis','oupneumo','reintub','pulembol','failwean','urninfec','cnscva','othdvt',
            'othsysep','othseshock']
    is_morb = 0
    for var in morb:
        # If one of the complications under morbidity occurred, then flag in morbidity column
        if row[var] == 1:
            is_morb = 1
    return is_morb

# get_bmi
def get_bmi(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe before BMI categories.
    Returns
    -------
    df : pandas dataframe
        dataframe with BMI categories.
    ------
    This function calculates BMI for an individual based on height and weight.
    """
    # Calculating BMI
    df['bmi'] = (703*df['weight'])/(df['height']**2)
    # Getting BMI class using helper function
    df['bmi_class'] = df['bmi'].apply(bmi_classes)
    # Getting dummy variables for each BMI class
    df = pd.get_dummies(df,columns=['bmi_class'],prefix=['bmi'])
    return df

# bmi_classes
def bmi_classes(bmi):
    """
    Parameters
    ----------
    bmi : float
        BMI.
    Returns
    -------
    str
        BMI category.
    -------
    This helper function creates BMI categories that match the user guide.
    """
    if bmi < 18.5:
        return 'underweight'
    elif bmi >= 18.5 and bmi < 25:
        return 'normal'
    elif bmi >= 25 and bmi < 30:
        return 'overweight'
    elif bmi >= 30 and bmi < 35:
        return 'obese_1'
    elif bmi >= 35 and bmi < 40:
        return 'obese_2'
    else:
        return 'obese_3'

# round_df
def round_df(df):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe to be rounded.
    Returns
    -------
    df : pandas dataframe
        rounded dataframe.
    -------
    This function rounds any values to match the user guide.
    """
    # Rounding imputed binary variables
    df = df.round({'sex':0,'ethnicity_hispanic':0,'stillinhosp':0,'ventilat':0,'hxcopd':0,
                   'ascites':0,'hxchf':0,'hypermed':0,'renafail':0,'dialysis':0,'discancr':0,
                   'wndinf':0,'steroid':0,'wtloss':0,'bleeddis':0,'transfus':0,'returnor':0})
    for column in df:
        # Rounding imputed PATOS binary variables
        if 'patos' in column:
            df = df.round({column:0})
        # Rounding continuous variables to the correct number of decimal places
        elif 'pr' in column:
            df = df.round({column:1})
        # Elective surgery and sex are special cases and employ a helper function
        elif column == 'electsurg':
            df.loc[:,'electsurg'] = df.loc[:,'electsurg'].apply(round_elec)
        elif column == 'sex':
            df.loc[:,'sex'] = df.loc[:,'sex'].apply(round_elec)
    return df

# round_elec 
def round_elec(elec):
    """
    Parameters
    ----------
    elec : float
        elective surgery value to be rounded after imputation.
    Returns
    -------
    float
        proper elective surgery value after mistakes fixed.
    -------
    This helper function is used for elec/sex as they are binary variables.
    """
    if elec <= 0:
        return 0.0
    elif elec > 0 and elec <= 0.5:
        return 0.0
    elif elec > 0.5 and elec <= 1:
        return 1.0
    elif elec >= 1:
        return 1.0
        
# clean
def clean():
    """
    Returns
    -------
    None.
    -------
    This function calls all of the others to take a raw NSQIP dataset and
    transform it into a clean csv file.
    """
    # Reading csv
    surg = pd.read_csv('data/nsqip_09_17.csv', encoding='latin1')
    
    # Options for displaying output in pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_colwidth', -1)
    
    # Cleaning
    surg = clean_df(surg)
    
    print(surg['ethnicity_hispanic'].unique())
    print(surg['transt'].unique())
    print(surg['dischdest'].unique())
    print(surg['surgspec'].unique())
    
    # Encoding
    surg = encode_vars(surg)
    
    print(surg.nunique())
    
    # Aggregating
    surg = aggregate_response(surg)
    
    # Getting BMI
    surg = get_bmi(surg)
    
    # Rounding
    surg = round_df(surg)
    
    # Output to csv
    surg.to_csv('data/surg.csv')
    
    # Seeing if the final list of variables is correct with right number of unique values
    f = open('columns.txt','w')
    print(surg.nunique(), file=f)
    f.close()

### Main ###
if __name__ == '__main__':
    clean()