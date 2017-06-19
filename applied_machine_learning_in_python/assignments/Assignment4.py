
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[ ]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer


def blight_model():
    # load data
    train_data_raw = pd.read_csv('train.csv', encoding='ISO-8859-1')
    grade_data_raw = pd.read_csv('test.csv')
    addresses = pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')

    # merge into one dataset
    grade_data_raw['compliance'] = np.nan
    grade_data_raw['grade_data'] = True
    train_data_raw['grade_data'] = False
    merged_data = train_data_raw.loc[
        ~train_data_raw['compliance'].isnull(), grade_data_raw.columns].append(
        grade_data_raw, ignore_index=True)

    # cast to correct formats
    merged_data['ticket_issued_date'] = pd.to_datetime(merged_data['ticket_issued_date'])
    merged_data['hearing_date'] = pd.to_datetime(merged_data['hearing_date'])

    # attach lat and lon
    addresses_latlons = pd.merge(addresses, latlons, on='address')
    all_data = pd.merge(merged_data, addresses_latlons, 'left', on='ticket_id')

    # add features
    all_data['ticket_month'] = all_data['ticket_issued_date'].dt.month.astype(str)
    all_data['ticket_day'] = all_data['ticket_issued_date'].dt.dayofweek.astype(str)
    all_data['hearing_month'] = all_data['hearing_date'].dt.month.astype(str)
    all_data['hearing_day'] = all_data['hearing_date'].dt.dayofweek.astype(str)
    all_data['days_between_ticket_and_hearing'] = (all_data['hearing_date'] - all_data[
        'ticket_issued_date']) / np.timedelta64(1, 'D')

    # choose features
    categoricals = ['agency_name', 'inspector_name', 'violation_code', 'disposition', 'violation_zip_code',
                    'ticket_month', 'hearing_month', 'hearing_day', 'ticket_day']
    numerics = ['days_between_ticket_and_hearing']

    # cast to correct formats
    for column in categoricals:
        all_data[column] = all_data[column].astype(str)
    for column in numerics:
        all_data[column] = all_data[column].astype(float)

    # create dummies and split data
    all_data_with_dummies = pd.get_dummies(all_data[categoricals + numerics])
    for column in ['compliance', 'grade_data', 'ticket_id']:
        all_data_with_dummies[column] = all_data[column]
    training_data = all_data_with_dummies[~all_data_with_dummies['grade_data']].drop(
        ['compliance', 'grade_data', 'ticket_id'], axis=1)
    training_compliance = all_data_with_dummies.loc[~all_data_with_dummies['grade_data'], 'compliance']
    grade_data = all_data_with_dummies[all_data_with_dummies['grade_data']].drop(['compliance', 'grade_data'], axis=1)

    # remove columns not applicable in grade_data and remove rows where there are no compliance applicable
    x_train, x_test, y_train, y_test = train_test_split(training_data, training_compliance)

    # handle nans
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(training_data)

    # train
    clf = LogisticRegression()
    clf.fit(imp.transform(x_train), y_train)

    # return predictions
    return pd.Series(
        [val[1] for val in clf.predict_proba(imp.transform(grade_data.drop('ticket_id', axis=1)))],
        index=grade_data['ticket_id'])


import pandas as pd
import numpy as np


def blight_model():

    from sklearn.model_selection import train_test_split

    train_data_raw = pd.read_csv('train.csv', encoding='ISO-8859-1')
    grade_data_raw = pd.read_csv('test.csv')
    addresses = pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')

    # cast to correct formats
    for df in [train_data_raw, grade_data_raw]:
        df['ticket_issued_date'] = pd.to_datetime(train_data_raw['ticket_issued_date'])
        df['hearing_date'] = pd.to_datetime(train_data_raw['hearing_date'])

    # attach lat and lon
    addresses_latlons = pd.merge(addresses, latlons, on='address')
    train_data_unfiltered = pd.merge(train_data_raw, addresses_latlons, 'left', on='ticket_id')
    grade_data = pd.merge(grade_data_raw, addresses_latlons, 'left', on='ticket_id')

    # add features
    for df in [train_data_unfiltered, grade_data]:
        df['ticket_month'] = df['ticket_issued_date'].map(lambda date: date.month)
        df['ticket_day'] = df['ticket_issued_date'].map(lambda date: date.day)
        df['hearing_month'] = df['hearing_date'].map(lambda date: date.month)
        df['hearing_day'] = df['hearing_date'].map(lambda date: date.day)
        df['days_between_ticket_and_hearing'] = (df['hearing_date'] - df['ticket_issued_date']) / np.timedelta64(1, 'D')

    # choose features
    categoricals = ['ticket_month', 'hearing_month']
    numerics = ['fine_amount', 'late_fee', 'discount_amount', 'judgment_amount', 'days_between_ticket_and_hearing',
                'hearing_day', 'ticket_day']

    # cast to correct formats
    for df in [train_data_unfiltered, grade_data]:
        for column in categoricals:
            df[column] = df[column].astype(str)
        for column in numerics:
            df[column] = df[column].astype(float)

    # filter data and omit features out of interest
    train_data_filtered = train_data_unfiltered.loc[
        ~train_data_unfiltered['compliance'].isnull(), categoricals + numerics + ['compliance']]
    train_data_with_dummies = pd.get_dummies(train_data_filtered)
    grade_data_with_dummies = pd.get_dummies(grade_data[categoricals + numerics])

    # remove columns not applicable in grade_data and remove rows where there are no compliance applicable
    x_train, x_test, y_train, y_test = train_test_split(
        train_data_with_dummies.drop('compliance', axis=1), train_data_with_dummies['compliance'], random_state=0)

    # handle nan
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)

    # train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0)
    clf.fit(imp.transform(x_train), y_train)
    clf.score(imp.transform(x_test), y_test)
    clf.predict_proba(imp.transform(x_test))


    return pd.Series([val[1] for val in clf.predict_proba(imp.transform(grade_data_with_dummies))],
                     index=grade_data['ticket_id'])