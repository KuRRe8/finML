import os
import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim

import lstm_def


"""
Step 0: Set param
"""
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'

_DEBUG = True                                               # 
_DEBUG_FALSE = False                                        # Always False

UNIQUE_THRESHOLD = 1                                      # for each feature, at least this number of unique values considered as good for discrimination, should be small number
FEATURE_SELECTION_RATIO = 0.1                               # for each feature, those process less than this ratio of values will be removed 0.7*400k=280k
YEAR_LIMIT = 3

TARGET_NAME = 'xrd'                                         # target column name
PRESERVE_LIST = []
ALWAYS_DROP_LIST = ['cashflow_v', 'market_value', 'yield', 'return_volatility', 'stock_price', 'GVKEY', 'sic']
pathprefix = os.path.dirname(os.path.abspath(__file__))
origin_path = None
with open(os.path.join(pathprefix,'..','datapath'), 'r') as f:
    origin_path = f.readline().strip()
METHODNAME = 'lstm'

logger = logging.getLogger()
filehandler = logging.FileHandler(os.path.join(pathprefix,'out', f'{METHODNAME}.log'))
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)

logger.info('Started.')

"""
Step 1: Data Cleaning
"""

origin = pd.read_csv(origin_path,low_memory=False)
logger.info(f"Data loaded from {origin_path}")
origin = origin.drop(columns=ALWAYS_DROP_LIST, errors='ignore')
origin.loc[origin[TARGET_NAME] < 0, TARGET_NAME] = 0
target_col = origin[TARGET_NAME] # always preserve target column

droplist = [col for col in origin.columns if origin[col].nunique() < UNIQUE_THRESHOLD]
origin.drop(droplist, axis=1, inplace=True)
if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s1alwaysdrop.csv'), index=False)

total_samples = origin.shape[0]
droplist = [col for col in origin.columns if origin[col].isnull().sum() > total_samples * (1 - FEATURE_SELECTION_RATIO)]

origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s2strongfeature.csv'), index=False)

# now we have much smaller dataset, 1 datetime and 19 object need to be processed
# object1: gvkey
s_gvkey = origin['gvkey'].copy()

# datetime641: datadate
base_date = pd.Timestamp('1987-01-01')
#origin['datadate'] = (pd.to_datetime(origin['datadate']) - base_date).dt.days

if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s3converttype.csv'), index=False)

# all other objects/strings dropped. tic, cusip, conm, cik, add2, weburl, etc.
droplist = [col for col in origin.columns if origin[col].dtype == 'object']
origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv('temp\\origin4.csv', index=False)
    origin.describe().to_csv('temp\\origin4_describe.csv')

#common_columns = origin.columns.intersection(preserve_cols.columns)
#origin = origin.drop(columns=common_columns)
#origin = pd.concat([origin, preserve_cols], axis=1)

if TARGET_NAME not in origin.columns:
    origin = pd.concat([origin, target_col], axis=1) # always preserve target column


imputer = SimpleImputer(strategy='median')
imputer_target_col = SimpleImputer(strategy='median')
scaler = preprocessing.StandardScaler()

origin.drop(TARGET_NAME, axis=1, inplace=True)

# impute excluding target column
origin_imputed = imputer.fit_transform(origin)
origin_scaled = scaler.fit_transform(origin_imputed)
origin_imputed = pd.DataFrame(origin_imputed, columns=origin.columns)

target_imputed = imputer_target_col.fit_transform(target_col.values.reshape(-1,1))

grouped_by_fyear = origin.groupby('gvkey',group_keys=True)

for key, group in grouped_by_fyear:
    if len(group) < YEAR_LIMIT:
        origin = origin[origin['gvkey'] != key]

grouped_by_fyear_1 = origin.groupby('gvkey',group_keys=True)
