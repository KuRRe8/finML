import os
import numpy as np
import pandas as pd
import enum
import logging

from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
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
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

_DEBUG = True                                               # 
_DEBUG_FALSE = False                                        # Always False

#UNIQUE_THRESHOLD = 3                                      # for each feature, at least this number of unique values considered as good for discrimination, should be small number
#FEATURE_SELECTION_RATIO = 0.7                               # for each feature, those process less than this ratio of values will be removed 0.7*400k=280k
#NEED_PCA = False                                            # whether to use PCA for feature selection, another point of view
#class DimReduction(enum.Enum):
#    None_ = 0
#    PCA = 1

TARGET_NAME = 'xrd'                                         # target column name from origin
ADDED_COLS = ['risk']                                       # features added outside
INTERESTED_ORIGIN_COLS = ['ppent','at','dlc','dltt','ib','dp','ceq','txdb','csho','prcc_f'] # features from origin

with open('originPath.txt', 'r', encoding='utf-8') as f:
    origin_path = f.readline().strip()
origin_path = os.path.expanduser(origin_path)

logger = logging.getLogger()
filehandler = logging.FileHandler('out\\linear_ridge.log')
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)

logger.info('Started.')

"""
Step 1: Data Cleaning
"""

origin = pd.read_stata(origin_path, preserve_dtypes=False)
origin.loc[origin[TARGET_NAME] < 0, TARGET_NAME] = 0
target_col = origin[TARGET_NAME] # always preserve target column
total_samples = origin.shape[0]

origin0 = None
origin1 = None
origin2 = origin[INTERESTED_ORIGIN_COLS + [TARGET_NAME]]

origin2.loc[:, 'ceq'] = origin2.loc[:, 'ceq'].fillna(0)
origin2.loc[:, 'txdb'] = origin2.loc[:, 'txdb'].fillna(0)
# devide prediction set and train_test set as early as possible
prediction_set2 = origin2[origin2[TARGET_NAME].isnull()]
#train_set = pd.concat([origin, prediction_set]).drop_duplicates(keep=False)
train_test_set2 = origin2[origin2[TARGET_NAME].notnull()]


if _DEBUG_FALSE:
    prediction_set2.info()
    train_test_set2.info()
    #prediction_set2.to_csv('temp1\\prediction5.csv', index=False)
    prediction_set2.describe().to_csv('temp1\\prediction5_describe.csv')
    #train_test_set2.to_csv('temp1\\train5.csv', index=False)
    train_test_set2.describe().to_csv('temp1\\train5_describe.csv')
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

X_train, X_test, y_train, y_test = train_test_split(train_test_set2.drop(columns=[TARGET_NAME]), train_test_set2[TARGET_NAME], test_size=0.25, random_state=42) #random state to make the result reproducible

the_pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values = pd.NA)),
    ('scaler', preprocessing.StandardScaler()),
    ('estimator', Ridge()) #analytical solution
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)


param_grid = {
    'imputer__strategy': ['mean', 'median', 'most_frequent'],
    'estimator__alpha': [0.01, 8.0, 10.0] + list(np.arange(0.1, 2.5, 0.1)),
    'estimator__fit_intercept': [True, False],
    'estimator__positive': [True, False],
}


grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'neg_mean_squared_error') # leave one core for other operations

grid_search.fit(X_train, y_train)

logger.info(f"Best alpha: {grid_search.best_params_}")
logger.info(f"Best cross-validated RMSE: {sqrt(-grid_search.best_score_)}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
logger.info(f"R2 score on test set: {r2}")


prediction_set2.loc[:,TARGET_NAME] = grid_search.predict(prediction_set2.drop(columns=[TARGET_NAME]))

prediction_set2[TARGET_NAME].to_csv('out\\explicit_2_linear_ridge.csv', index=False)
##prediction_set.to_csv('out\\linear_ridge_all.csv', index=False)