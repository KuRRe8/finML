import os
import numpy as np
import pandas as pd
import enum
import logging

from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.feature_extraction
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, cross_validate
from math import sqrt

"""
Step 0: Set param
"""
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'

_DEBUG = True                                               # 
_DEBUG_FALSE = False                                        # Always False

UNIQUE_THRESHOLD = 1                                      # for each feature, at least this number of unique values considered as good for discrimination, should be small number
FEATURE_SELECTION_RATIO = 0.1                               # for each feature, those process less than this ratio of values will be removed 0.7*400k=280k
NEED_PCA = False                                            # whether to use PCA for feature selection, another point of view
class DimReduction(enum.Enum):
    None_ = 0
    PCA = 1

TARGET_NAME = 'xrd'                                         # target column name
PRESERVE_LIST = []
ALWAYS_DROP_LIST = ['cashflow_v', 'market_value', 'yield', 'return_volatility', 'stock_price', 'GVKEY', 'sic']
pathprefix = os.path.dirname(os.path.abspath(__file__))
origin_path = None
with open(os.path.join(pathprefix,'datapath'), 'r') as f:
    origin_path = f.readline().strip()
METHODNAME = 'linear_OLS'

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
hasher = sklearn.feature_extraction.FeatureHasher(n_features=2**3, input_type='string') 
s_gvkey = s_gvkey.astype('category')
origin['gvkey'] = s_gvkey
hashed_features = hasher.transform(origin['gvkey'].astype('string').apply(lambda x: [x]))
hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'gvkey_hash_{i}' for i in range(hashed_features.shape[1])])
origin = pd.concat([origin, hashed_df], axis=1)
#origin.drop('gvkey', axis=1, inplace=True)

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

# devide prediction set and train_test set as early as possible

prediction_set = origin[origin[TARGET_NAME].isnull()]
prediction_set = prediction_set.drop('gvkey', axis=1)

filtered_indices = []
def find_non_empty_key2(group: pd.DataFrame) -> None:
    global filtered_indices
    preceedingNa = False
    for index, row in group.iterrows():
        if pd.notna(row[TARGET_NAME]):
            if preceedingNa:
                filtered_indices.append(index)
            preceedingNa = False
        else:
            preceedingNa = True

origin.groupby('gvkey').apply(find_non_empty_key2, include_groups=False)

test_set = origin.loc[filtered_indices].copy().drop('gvkey', axis=1, errors='ignore')
train_set = origin.drop(filtered_indices).copy()
train_set.drop('gvkey', axis=1, inplace=True, errors='ignore')
train_set = train_set[train_set[TARGET_NAME].notnull()]


X_train = train_set.drop(columns=[TARGET_NAME])
y_train = train_set[TARGET_NAME]
X_test = test_set.drop(columns=[TARGET_NAME])
y_test = test_set[TARGET_NAME]


'''
# standardize the data, process 400k samples together
scaler = preprocessing.StandardScaler().fit(imputed.to_numpy())

step1_final = pd.DataFrame(scaler.transform(imputed.to_numpy()), columns=imputed.columns)

if NEED_PCA:
    NotImplementedError('PCA not implemented yet')
else:
    pass
'''

for i in range(1, 2):
    #X_train, X_test, y_train, y_test = train_test_split(train_test_set.drop(columns=[TARGET_NAME]), train_test_set[TARGET_NAME], test_size=0.25, random_state=None) #random state to make the result reproducible

    imputer = SimpleImputer(strategy='median')
    scaler = preprocessing.StandardScaler()
    estimator = LinearRegression(positive=True)

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    estimator.fit(X_train, y_train)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    y_pred = estimator.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"R2 score on test set: {r2}")
    logger.info(f"RMSE on test set: {rmse}")



    _prediction_set = prediction_set.drop(columns=[TARGET_NAME])
    _prediction_set = imputer.transform(_prediction_set)
    _prediction_set = scaler.transform(_prediction_set)
    
    prediction_set.loc[:,TARGET_NAME] = estimator.predict(_prediction_set)

    prediction_set.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}.csv'), index=False)

    origin = pd.concat([origin, s_gvkey], axis=1)
    origin.loc[origin[TARGET_NAME].isnull(), TARGET_NAME] = prediction_set[TARGET_NAME]
    origin.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}_full.csv'), index=False)