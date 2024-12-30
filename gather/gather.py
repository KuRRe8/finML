# 1 - lasso
# 2 - ridge
# 3 - OLS
# 4 - MLP
# 5 - RF
# 6 - XGB


import sys
# Ensure the input is valid
if len(sys.argv) < 2:
    raise ValueError("Please provide a list of tasks as command line arguments.")

# Convert input arguments to a list of integers
tasks = list(map(int, sys.argv[1:]))

# Validate the tasks
if not all(1 <= task <= 6 for task in tasks):
    raise ValueError("Tasks must be integers between 1 and 6.")
if len(tasks) != len(set(tasks)):
    raise ValueError("Each task must be unique.")




UNIQUE_THRESHOLD = 1                                      # for each feature, at least this number of unique values considered as good for discrimination, should be small number
FEATURE_SELECTION_RATIO = 0.999                               # for each feature, those process less than this ratio of values will be removed 0.7*400k=280k

TARGET_NAME_COLS = ['xrd', 'logxrd', 'xrdat', 'xrdppe', 'xrdrank']
PRESERVE_LIST = []
ALWAYS_DROP_LIST = ['cashflow_v', 'market_value', 'yield', 'return_volatility', 'stock_price', 'GVKEY', 'sic']
ALWAYS_DROP_LIST = []


_DEBUG = True                                               # 
_DEBUG_FALSE = False                                        # Always False

METHODNAME = ['lasso', 'ridge', 'OLS', 'MLP', 'RF', 'XGB']

import os
import numpy as np
import pandas as pd
import enum
import logging

from sklearn.linear_model import Ridge, Lasso, LinearRegression
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
from xgboost import XGBRegressor
from xgboost import DMatrix
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime


"""
Step 0: Set param
"""


os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'
pathprefix = os.path.dirname(os.path.abspath(__file__))
origin_path = None
with open(os.path.join(pathprefix,'..','datapath'), 'r') as f:
    origin_path = f.readline().strip()


logger = logging.getLogger()
log_filename = f"gather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
filehandler = logging.FileHandler(os.path.join(pathprefix, 'out', log_filename))
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)

logger.info('Started.')

"""
Step 1: Data Cleaning
"""
origin = pd.read_stata(origin_path)
logger.info(f"Data loaded from {origin_path}")
origin = origin.drop(columns=ALWAYS_DROP_LIST, errors='ignore')

target_cols = [col for col in origin.columns if col in TARGET_NAME_COLS]
target_df = origin[target_cols] # preserve 5 target columns

droplist = [col for col in origin.columns if origin[col].nunique() < UNIQUE_THRESHOLD]
origin.drop(droplist, axis=1, inplace=True)
if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s1alwaysdrop.csv'), index=False)

total_samples = origin.shape[0]
droplist = [col for col in origin.columns if origin[col].isnull().sum() > total_samples * (1 - FEATURE_SELECTION_RATIO)]

origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s2strongfeature.csv'), index=False)


# process gvkey, still keep the original gvkey
s_gvkey = origin['gvkey'].copy()
hasher = sklearn.feature_extraction.FeatureHasher(n_features=2**3, input_type='string') 
s_gvkey = s_gvkey.astype('category')
origin['gvkey'] = s_gvkey
hashed_features = hasher.transform(origin['gvkey'].astype('string').apply(lambda x: [x]))
hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'gvkey_hash_{i}' for i in range(hashed_features.shape[1])])
origin = pd.concat([origin, hashed_df], axis=1)

# datetime641: datadate
base_date = pd.Timestamp('1987-01-01')
origin['datadate'] = (pd.to_datetime(origin['datadate']) - base_date).dt.days



if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s3converttype.csv'), index=False)

# all other objects/strings dropped. tic, cusip, conm, cik, add2, weburl, etc.
droplist = [col for col in origin.columns if origin[col].dtype == 'object']
origin.drop(droplist, axis=1, inplace=True)

if _DEBUG_FALSE:
    origin.to_csv(os.path.join('temp','origin4.csv'), index=False)
    origin.describe().to_csv(os.path.join('temp','origin4_describe.csv'))

# merge target columns back
origin = pd.concat([origin, target_df], axis=1) # always preserve target column


curr_col = ''
X_train = []
y_train = []
X_test_sub = []
y_test_sub = []
X_test = []
y_test = []

for ind in range(len(TARGET_NAME_COLS)):
    print(f'Processing {TARGET_NAME_COLS[ind]}')
    #prediction_set = origin[origin[TARGET_NAME_COLS[ind]].isnull()]

    train_test_set = origin[origin[TARGET_NAME_COLS[ind]].notnull()]


    filtered_indices = []
    def find_non_empty_key2(group: pd.DataFrame) -> None:
        global filtered_indices
        preceedingNa = False
        for index, row in group.iterrows():
            if pd.notna(row[TARGET_NAME_COLS[ind]]):
                if row['fyear'] <= 2011:
                    return
                else:
                    filtered_indices.append(index)
                    return

    origin.groupby('gvkey', observed=False).apply(find_non_empty_key2, include_groups=False)

    subsample_test_set = origin.loc[filtered_indices].copy().drop('gvkey', axis=1, errors='ignore')
    subsample_test_set.dropna(inplace=True)
    fullsample_test_set = train_test_set[train_test_set['fyear'] > 2011].copy()
    fullsample_test_set.drop('gvkey', axis=1, inplace=True, errors='ignore')
    train_set = train_test_set[train_test_set['fyear'] < 2012].copy()
    train_set.drop('gvkey', axis=1, inplace=True, errors='ignore')

    X_train.append(train_set.drop(columns=TARGET_NAME_COLS))
    y_train.append(train_set[TARGET_NAME_COLS[ind]]) # target_cols is a list of 5 columns
    X_test_sub.append(subsample_test_set.drop(columns=TARGET_NAME_COLS))
    y_test_sub.append(subsample_test_set[TARGET_NAME_COLS[ind]])
    X_test.append(fullsample_test_set.drop(columns=TARGET_NAME_COLS))
    y_test.append(fullsample_test_set[TARGET_NAME_COLS[ind]])

logger.info('Data cleaning finished.')

if _DEBUG_FALSE:
    for i in range(len(y_test)):
        print(y_test[i].describe())
        y_test[i].describe().to_csv(os.path.join(pathprefix,'out',f'y_test{i}_describe.csv'))
    
    exit(0)

"""
Step 2: Training
"""

for task in tasks:

    if task == 1: #lasso
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")
            the_pipe = Pipeline([
                ('imputer', SimpleImputer(missing_values = pd.NA)),
                ('scaler', preprocessing.StandardScaler()),
                ('estimator', Lasso()) #iterative solution
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=None)


            param_grid = {
                'imputer__strategy': ['mean', 'median', 'most_frequent'],
                'estimator__alpha': [0.01, 8.0, 10.0] + list(np.arange(0.1, 3.1, 0.1)),
                'estimator__fit_intercept': [True, False],
                'estimator__positive': [True, False],
            }


            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

            grid_search.fit(X_train[ind], y_train[ind])

            logger.info(f"Best alpha: {grid_search.best_params_}")
            logger.info(f"Best cross-validated R2: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_


            y_pred = best_model.predict(X_test[ind])

            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            r2 = r2_score(y_test[ind], y_pred)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")

            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")


            y_pred = best_model.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            r2 = r2_score(y_test_sub[ind], y_pred)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")

            #prediction_set.loc[:,TARGET_NAME] = grid_search.predict(prediction_set.drop(columns=[TARGET_NAME]))

            #prediction_set.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}.csv'), index=False)

            #origin = pd.concat([origin, s_gvkey], axis=1)
            #origin.loc[origin[TARGET_NAME].isnull(), TARGET_NAME] = prediction_set[TARGET_NAME]
            #origin.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}_full.csv'), index=False)
    elif task == 2: #ridge
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")
            the_pipe = Pipeline([
                ('imputer', SimpleImputer(missing_values = pd.NA)),
                ('scaler', preprocessing.StandardScaler()),
                ('estimator', Ridge()) #analytical solution
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=None)


            param_grid = {
                'imputer__strategy': ['mean', 'median', 'most_frequent'],
                'estimator__alpha': [0.01, 8.0, 10.0] + list(np.arange(0.1, 2.5, 0.1)),
                'estimator__fit_intercept': [True, False],
                'estimator__positive': [True, False],
            }


            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

            grid_search.fit(X_train[ind], y_train[ind])

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best cross-validated R2: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test[ind])

            r2 = r2_score(y_test[ind], y_pred)
            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")
            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")


            y_pred = best_model.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            r2 = r2_score(y_test_sub[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")


    elif task == 3: #OLS
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")
            imputer = SimpleImputer(strategy='median')
            scaler = preprocessing.StandardScaler()
            estimator = LinearRegression(positive=True)

            X_train[ind] = imputer.fit_transform(X_train[ind])
            X_train[ind] = scaler.fit_transform(X_train[ind])
            estimator.fit(X_train[ind], y_train[ind])

            X_test[ind] = imputer.transform(X_test[ind])
            X_test[ind] = scaler.transform(X_test[ind])
            y_pred = estimator.predict(X_test[ind])

            r2 = r2_score(y_test[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")
            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")



            y_pred = estimator.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            r2 = r2_score(y_test_sub[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")




    elif task == 4: #MLP
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")

            the_pipe = Pipeline([
                ('imputer', SimpleImputer(missing_values = pd.NA)),
                ('scaler', preprocessing.StandardScaler()),
                ('estimator', MLPRegressor()) #search solution
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=None)


            param_grid = {
                'imputer__strategy': ['median'],
                'estimator__activation': [ 'relu', 'sigmoid'],
                'estimator__solver': ['sgd', 'adam'],
                'estimator__alpha': [0.001, 0.05, 0.1, 0.5],
                'estimator__learning_rate': ['adaptive'],
                'estimator__verbose': [False],
                'estimator__max_iter': [1000]

            }


            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

            grid_search.fit(X_train[ind], y_train[ind])

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best cross-validated R2: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test[ind])

            r2 = r2_score(y_test[ind], y_pred)
            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")
            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")



            y_pred = best_model.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            r2 = r2_score(y_test_sub[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")


    elif task == 5: #RF
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")

            the_pipe = Pipeline([
                ('imputer', SimpleImputer(missing_values = pd.NA)),
                ('scaler', preprocessing.StandardScaler()),
                ('estimator', RandomForestRegressor()) #search solution
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=None)


            param_grid = {
                'imputer__strategy': ['mean'],
                'estimator__n_estimators': [10, 100],
                'estimator__max_depth': [10, 100],
                'estimator__min_samples_split': [ 10, 20],
                'estimator__min_samples_leaf': [10, 20],

            }


            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

            grid_search.fit(X_train[ind], y_train[ind])

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best cross-validated R2: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test[ind])

            r2 = r2_score(y_test[ind], y_pred)
            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")
            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")



            y_pred = best_model.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            r2 = r2_score(y_test_sub[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")


    elif task == 6: #XGB
        for ind in range(len(TARGET_NAME_COLS)):
            logger.info(f"Task {task} for {TARGET_NAME_COLS[ind]}")
            
            def custom_objective(y_true, y_pred):
                grad = np.where(y_pred < 0, 2 * (y_pred - y_true) * 2, 2 * (y_pred - y_true))
                hess = np.where(y_pred < 0, 4, 2)
                return grad, hess
            

            the_pipe = Pipeline([
                ('imputer', SimpleImputer(missing_values = pd.NA)),
                ('scaler', preprocessing.StandardScaler()),
                ('estimator', XGBRegressor(tree_method='hist', device='cuda', objective=custom_objective)) 
            ])

            kf = KFold(n_splits=15, shuffle=True, random_state=None)


            param_grid = {
                'imputer__strategy': ['mean'],
                'estimator__n_estimators': [50, 170],
                'estimator__max_depth': [3, 11],
                'estimator__learning_rate': [0.01, 0.3],
                'estimator__subsample': [0.5, 1]
            }


            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

            grid_search.fit(X_train[ind], y_train[ind])

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best cross-validated R2: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test[ind])

            r2 = r2_score(y_test[ind], y_pred)
            n = X_test[ind].shape[0]  # number of samples
            p = X_test[ind].shape[1]  # number of features
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            rmse = sqrt(mean_squared_error(y_test[ind], y_pred))
            logger.info(f"R2 score on test set: {r2}")
            logger.info(f"RMSE on test set: {rmse}")
            logger.info(f"Adjusted R2 score on test set: {adjusted_r2}")



            y_pred = best_model.predict(X_test_sub[ind])

            n = X_test_sub[ind].shape[0]  # number of samples
            p = X_test_sub[ind].shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            r2 = r2_score(y_test_sub[ind], y_pred)
            rmse = sqrt(mean_squared_error(y_test_sub[ind], y_pred))
            logger.info(f"R2 score on sudden report subsamples test set: {r2}")
            logger.info(f"RMSE on sudden report subsamples test set: {rmse}")
            logger.info(f"Adjusted R2 score on sudden report subsamples test set: {adjusted_r2}")

logger.info('Training finished.')