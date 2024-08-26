import os
import numpy as np
import pandas as pd
import enum
import logging

from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, BayesianRidge
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

UNIQUE_THRESHOLD = 3                                      # for each feature, at least this number of unique values considered as good for discrimination, should be small number
FEATURE_SELECTION_RATIO = 0.7                               # for each feature, those process less than this ratio of values will be removed 0.7*400k=280k
NEED_PCA = False                                            # whether to use PCA for feature selection, another point of view
class DimReduction(enum.Enum):
    None_ = 0
    PCA = 1

TARGET_NAME = 'xrd'                                         # target column name

origin_path = os.path.expanduser('~/Downloads/annfunda.dta')

logger = logging.getLogger()
filehandler = logging.FileHandler('out\\linear_bayesian.log')
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)

logger.info('Started.')

"""
Step 1: Data Cleaning
"""

origin = pd.read_stata(origin_path, preserve_dtypes=False)

target_col = origin[TARGET_NAME] # always preserve target column

droplist = [col for col in origin.columns if origin[col].nunique() < UNIQUE_THRESHOLD]
origin.drop(droplist, axis=1, inplace=True)
if _DEBUG_FALSE:
    origin.to_csv('temp\\origin1.csv', index=False)

total_samples = origin.shape[0]
droplist = [col for col in origin.columns if origin[col].isnull().sum() > total_samples * (1 - FEATURE_SELECTION_RATIO)]

origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv('temp\\origin2.csv', index=False)

#origin.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 401947 entries, 0 to 401946
#Columns: 175 entries, gvkey to weburl
#dtypes: datetime64[ns](1), float64(155), object(19)
#memory usage: 536.7+ MB

# now we have much smaller dataset, 1 datetime and 19 object need to be processed
# object1: gvkey
s_gvkey = origin['gvkey']
#hasher = sklearn.feature_extraction.FeatureHasher(n_features=2**18, input_type='string') #use 256K hash space for 37.4K distinct gvkey
s_gvkey = s_gvkey.astype('int64')
origin['gvkey'] = s_gvkey

# datetime641: datadate
base_date = pd.Timestamp('1987-01-01')
origin['datadate'] = (origin['datadate'] - base_date).dt.days

if _DEBUG_FALSE:
    origin.to_csv('temp\\origin3.csv', index=False)

# all other objects/strings dropped. tic, cusip, conm, cik, add2, weburl, etc.
droplist = [col for col in origin.columns if origin[col].dtype == 'object']
origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv('temp\\origin4.csv', index=False)
    origin.describe().to_csv('temp\\origin4_describe.csv')

if TARGET_NAME not in origin.columns:
    origin = pd.concat([origin, target_col], axis=1) # always preserve target column

# devide prediction set and train_test set as early as possible
prediction_set = origin[origin[TARGET_NAME].isnull()]
#train_set = pd.concat([origin, prediction_set]).drop_duplicates(keep=False)
train_test_set = origin[origin[TARGET_NAME].notnull()]

if _DEBUG_FALSE:
    prediction_set.info()
    train_test_set.info()
    prediction_set.to_csv('temp1\\prediction5.csv', index=False)
    prediction_set.describe().to_csv('temp1\\prediction5_describe.csv')
    train_test_set.to_csv('temp1\\train5.csv', index=False)
    train_test_set.describe().to_csv('temp1\\train5_describe.csv')
    exit(0)  

if _DEBUG_FALSE: # should fit train set not origin or train_test_set!!!! 
    # impute missing values,
    myImputer = SimpleImputer(missing_values = pd.NA, strategy='mean')
    imputed = pd.DataFrame(myImputer.fit_transform(origin), columns= origin.columns)
    raise NotImplementedError('impute missing values should be done after split')
    if _DEBUG_FALSE:
        imputed.info()
        imputed.to_csv('temp\\imputed5.csv', index=False)
        imputed.describe().to_csv('temp\\imputed5_describe.csv')  

'''
# standardize the data, process 400k samples together
scaler = preprocessing.StandardScaler().fit(imputed.to_numpy())

step1_final = pd.DataFrame(scaler.transform(imputed.to_numpy()), columns=imputed.columns)

if NEED_PCA:
    NotImplementedError('PCA not implemented yet')
else:
    pass
'''

X_train, X_test, y_train, y_test = train_test_split(train_test_set.drop(columns=[TARGET_NAME]), train_test_set[TARGET_NAME], test_size=0.25, random_state=42) #random state to make the result reproducible

the_pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values = pd.NA)),
    ('scaler', preprocessing.StandardScaler()),
    ('estimator', BayesianRidge()) #iterative solution
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)


param_grid = {
    'imputer__strategy': ['mean', 'median', 'most_frequent'],
    'estimator__alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
    'estimator__alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
    'estimator__lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
    'estimator__lambda_2': [1e-6, 1e-5, 1e-4, 1e-3],
    'estimator__fit_intercept': [True, False],

}


grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

grid_search.fit(X_train, y_train)

logger.info(f"Best params: {grid_search.best_params_}")
logger.info(f"Best cross-validated R²: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred).round(4)
rmse = sqrt(mean_squared_error(y_test, y_pred))
logger.info(f"R2 score on test set: {r2}")
logger.info(f"RMSE on test set: {rmse}")





prediction_set.loc[:,TARGET_NAME] = grid_search.predict(prediction_set.drop(columns=[TARGET_NAME]))

prediction_set[TARGET_NAME].to_csv('out\\linear_bayesian.csv', index=False)
##prediction_set.to_csv('out\\linear_bayesian_all.csv', index=False)