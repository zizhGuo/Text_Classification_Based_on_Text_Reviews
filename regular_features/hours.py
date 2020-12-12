import numpy as np
import pandas as pd
import sys
import csv
import json
import string
import datetime
from csvsort import csvsort
import matplotlib.pyplot as plt

'''
"hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }

feature_creation_breakHours():
    Takes an dataframe and apply string operation on every sample's 'hours' feature
    Takes an 'hours' object and break its sub-object into 14 new individual features.

    'm1': Monday opening time
    'm2': Monday closing time
    't1': Tunesday opening time
    't2': Tunesday closing time
    'w1': Wednesday opening time
    'w2': Wednesday closing time
    'th1': Thursday opening time
    'th2': Thursday closing time
    'f1': Friday opening time
    'f2': Friday closing time
    's1': Saturday opening time
    's2': Saturday closing time
    'su1': Sunday opening time
    'su2': Sunday closing time

'''
def feature_creation_breakHours(df):
    hour = df['hours']
    dicts = hour.apply(lambda x : x.replace('\'','\"')).apply(json.loads)
    # dicts = df['hours']
    keys = ['Monday', 
            'Tuesday',  
            'Wednesday', 
            'Thursday',
            'Friday', 
            'Saturday', 
            'Sunday']
    # hours = [[int(time[0]) for y in keys for time in pd.Series(x[y].split('-')).apply(lambda x : x.split(':'))] for x in dicts]
    hours = []
    for d in dicts:
        ts = [d[y].split('-') if y in d.keys() else ['-10:00', '-34:00'] for y in keys]
        # ts : [['10:0', '1:0'], ['10:0', '1:0']...]
        res = pd.Series([int(y) for t in ts for x in t for y in x.split(':')])
        res_new = [(res[2 * i]*100 + res[2 *i + 1])/100 for i in range(0, 14)]
        hours.append(res_new)
    for hour in hours:
        for i in range(0, 7):
            if hour[2 * i + 1] < hour[2 * i]:
                hour[2 * i + 1] += 24
    df[['m1', 'm2', 't1', 't2', 'w1', 'w2', 'th1', 'th2', 'f1', 'f2', 's1', 's2', 'su1','su2']] = pd.DataFrame(hours, index=df.index)
    df = df.drop(columns = ['hours'])
    return df

def feature_creation_addMore(df):
    '''
        add new features:
        is24hoursAny
        isMondayOpen
        isSundayOpen
        weeklyBusinessHoursTotal
        weeklyBusinessHoursAverage
    '''
    df['is24hoursAny'] = (df['m1'] == 0) * (df['m2'] == 0) \
        | (df['t1'] == 0) * (df['t2'] == 0) \
        | (df['w1'] == 0) * (df['w2'] == 0) \
        | (df['th1'] == 0) * (df['th2'] == 0) \
        | (df['f1'] == 0) * (df['f2'] == 0) \
        | (df['s1'] == 0) * (df['s2'] == 0) \
        | (df['su1'] == 0) * (df['su2'] == 0)
    df['is24hoursAny'] = df['is24hoursAny'].astype(int)
    df['isMondayOpen'] = (df['m1'] < 0).astype(int)
    df['isSundayOpen'] = (df['su1'] < 0).astype(int)

    total_hours_list = []
    total_days_list = []
    for i in range(0, len(df)):
        total_hours = 0
        number_days = 7
        for j in range(1, 8):
            if df.iloc[i, 2* j] < 0:
                number_days -= 1
            if df.iloc[i, 2* j] >= df.iloc[i, 2* j + 1]:
                total_hours += df.iloc[i, 2* j + 1] + 24 - df.iloc[i, 2* j]
            else:
                total_hours += df.iloc[i, 2* j + 1] - df.iloc[i, 2* j]
        total_hours_list.append(total_hours)
        total_days_list.append(number_days)
    
    df['weeklyBusinessHoursTotal'] = total_hours_list
    df['weeklyBusinessDaysTotal'] = total_days_list
    df['weeklyBusinessHoursAverage'] = df['weeklyBusinessHoursTotal'] / df['weeklyBusinessDaysTotal']

    # new features added after anaylize the CCwith targets
    df['monday_new'] = df['s1'] - df['isMondayOpen'] * 8
    df['sunday_new'] = df['su1']  - df['isSundayOpen'] * 8

    return df

def feature_selection_fval(X, y, alpha = 0.5):
    """ 
    This is a feature selection function based on Sklearn library
        Params:
            @X: a list of list featured samples
            @y: a list of corresponding target values
            @alpha: a params that controls how much feature user wants to select
        Return:
            new featured samples
    """
    # if X.shape:
    #     n = int(np.floor(X.shape[1] * alpha))
    # else:
    n = int(np.floor(len(X[0]) * alpha))
    from sklearn.feature_selection import SelectKBest,f_classif
    sel=SelectKBest(score_func=f_classif,k=n)
    sel.fit(X,y)
    # print('scores_:\n',sel.scores_)
    # print('pvalues_:',sel.pvalues_)
    # print('selected index:',sel.get_support(True))
    # print('after transform:\n',sel.transform(X))
    X_new = sel.transform(X)
    return X_new

def get_regular_features(df):
    """ 
    This functions extracts the regular 'hours' feature into 22 independent features
        Params:
            @df: the dataframe contains 'hours' featured samples
        Return:
            extracted/engineered featured samples
    """
    # depends on the architecture of the dataset
    df = df.drop(columns = ['Unnamed: 0', 'stars', 'text', 'category']) 
    # X = df['hours']
    # print(X[:10])
    df_extracted = fe.feature_creation_breakHours(df) # 生成14个独立feature
    df_extracted = fe.feature_creation_addMore(df_extracted) # 生成14个独立feature
    from sklearn.preprocessing import MinMaxScaler
    df_extracted = MinMaxScaler().fit_transform(df_extracted)
    return df_extracted