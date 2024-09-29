import os
import numpy as np
import pandas as pd
import enum
import logging

from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.svm import SVR
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
INTERESTED_COLS_0 = ['tang','at','lev','cashflow','q','risk']
INTERESTED_COLS_1 = INTERESTED_COLS_0
INTERESTED_ORIGIN_COLS = ['ppent','at','dlc','dltt','ib','dp','ceq','txdb','csho','prcc_f'] # features from origin

with open('originPath.txt', 'r', encoding='utf-8') as f:
    origin_path = f.readline().strip()
origin_path = os.path.expanduser(origin_path)

logger = logging.getLogger()
filehandler = logging.FileHandler('out\\highDim_SVR.log')
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

origin = origin[INTERESTED_ORIGIN_COLS + [TARGET_NAME]] # no 'risk' yet
origin.loc[:, 'ceq'] = origin.loc[:, 'ceq'].fillna(0)
origin.loc[:, 'txdb'] = origin.loc[:, 'txdb'].fillna(0)
"""
do something to add 'risk' column
"""

myImputer = SimpleImputer(missing_values = pd.NA, strategy='mean')

origin0 = pd.DataFrame(myImputer.fit_transform(origin[INTERESTED_ORIGIN_COLS]), columns=INTERESTED_ORIGIN_COLS) # to be discussed
origin0[TARGET_NAME] = target_col
origin0['tang'] = (origin0['ppent'] / origin0['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin0['lev'] = ((origin0['dlc'] + origin0['dltt']) / origin0['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin0['cashflow'] = ((origin0['ib'] + origin0['dp'])/ origin0['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
#q=(at+csho*prcc_f-ceq-txdb)/at
origin0['q'] = ((origin0['at'] + origin0['csho'] * origin0['prcc_f'] - origin0['ceq'] - origin0['txdb']) / origin0['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin0['risk'] = 0.0

origin1 = pd.DataFrame(myImputer.fit_transform(origin[INTERESTED_ORIGIN_COLS]), columns=INTERESTED_ORIGIN_COLS) # to be discussed
origin1[TARGET_NAME] = target_col
origin1['tang'] = (origin1['ppent'] / origin1['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin1['lev'] = ((origin1['dlc'] + origin1['dltt']) / origin1['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin1['cashflow'] = ((origin1['ib'] + origin1['dp'])/ origin1['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
#q=(at-seq+csho*prcc_f)/at
origin1['q'] = ((origin1['at'] - origin1['ceq'] + origin1['csho'] * origin1['prcc_f']) / origin1['at']).replace([np.inf, -np.inf], np.nan).fillna(0)
origin1['risk'] = 0.0

origin0 = origin0[INTERESTED_COLS_0 + [TARGET_NAME]]
origin1 = origin1[INTERESTED_COLS_1 + [TARGET_NAME]]
origin2 = origin[INTERESTED_ORIGIN_COLS + [TARGET_NAME]]


prediction_set0 = origin0[origin0[TARGET_NAME].isnull()]
train_test_set0 = origin0[origin0[TARGET_NAME].notnull()]

prediction_set1 = origin1[origin1[TARGET_NAME].isnull()]
train_test_set1 = origin1[origin1[TARGET_NAME].notnull()]

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
while(True):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(train_test_set2.drop(columns=[TARGET_NAME]), train_test_set2[TARGET_NAME], test_size=0.25, random_state=42) #random state to make the result reproducible

    the_pipe2 = Pipeline([
        ('imputer2', SimpleImputer(missing_values = pd.NA)),
        ('scaler2', preprocessing.StandardScaler()),
        ('estimator2', SVR())
    ])

    kf2 = KFold(n_splits=5, shuffle=True, random_state=42)


    param_grid2 = {
    'imputer2__strategy': ['mean'],
    'estimator2__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'estimator2__C': [0.5, 1, 2],
    'estimator2__epsilon': [0.1, 0.2],
}



    grid_search2 = GridSearchCV(the_pipe2, param_grid2, cv=kf2, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

    grid_search2.fit(X_train2, y_train2)

    logger.info(f"Best param for our approach2: {grid_search2.best_params_}")
    logger.info(f"Best cross-validated R2 for our approach2: {grid_search2.best_score_}")

    best_model2 = grid_search2.best_estimator_
    y_pred2 = best_model2.predict(X_test2)

    r2 = r2_score(y_test2, y_pred2)
    logger.info(f"R2 score on test set for our approach2: {r2}")
    rmse = sqrt(mean_squared_error(y_test2, y_pred2))
    logger.info(f"RMSE on test set for our approach2: {rmse}")


    prediction_set2.loc[:,TARGET_NAME] = grid_search2.predict(prediction_set2.drop(columns=[TARGET_NAME]))

    prediction_set2[TARGET_NAME].to_csv('out\\approach2_highDim_SVR.csv', index=False)
    prediction_set2.to_csv('out\\approach2_highDim_SVR_all.csv', index=False)

    break

while True:

    """
    at this point, the train_test_set should be already imputed either with mode, median, mean, or zero
    the shape also should be adapted
    no more arithmatic, no more shape conversion
    we may use SimpleImputer.fit_transform() to the whole dataset, but it doesnt matter
    """

    X_train1, X_test1, y_train1, y_test1 = train_test_split(train_test_set1.drop(columns=[TARGET_NAME]), train_test_set1[TARGET_NAME], test_size=0.25, random_state=42) #random state to make the result reproducible
    # the X_test1 and y_test1 are not used in the training process, they are used to evaluate the model
    
    the_pipe1 = Pipeline([
        ('scaler1', preprocessing.StandardScaler()),
        ('estimator1', SVR())
    ])

    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)


    param_grid1 = {
    'estimator1__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'estimator1__C': [0.5, 1, 2],
    'estimator1__epsilon': [0.1, 0.2],
}



    grid_search1 = GridSearchCV(the_pipe1, param_grid1, cv=kf1, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

    grid_search1.fit(X_train1, y_train1)

    logger.info(f"Best param for our approach1: {grid_search1.best_params_}")
    logger.info(f"Best cross-validated R2 for our approach1: {grid_search1.best_score_}")

    best_model1 = grid_search1.best_estimator_
    y_pred1 = best_model1.predict(X_test1)

    r2 = r2_score(y_test1, y_pred1)
    logger.info(f"R2 score on test set for our approach1: {r2}")
    rmse = sqrt(mean_squared_error(y_test1, y_pred1))
    logger.info(f"RMSE on test set for our approach1: {rmse}")

    prediction_set1.loc[:,TARGET_NAME] = grid_search1.predict(prediction_set1.drop(columns=[TARGET_NAME]))

    prediction_set1[TARGET_NAME].to_csv('out\\approach1_highDim_SVR.csv', index=False)
    prediction_set1.to_csv('out\\approach1_highDim_SVR_all.csv', index=False)

    
    break

while True:
    """
    at this point, the train_test_set should be already imputed either with mode, median, mean, or zero
    the shape also should be adapted
    no more arithmatic, no more shape conversion
    we may use SimpleImputer.fit_transform() to the whole dataset, but it doesnt matter
    """

    X_train0, X_test0, y_train0, y_test0 = train_test_split(train_test_set0.drop(columns=[TARGET_NAME]), train_test_set0[TARGET_NAME], test_size=0.25, random_state=42) #random state to make the result reproducible
    # the X_test0 and y_test0 are not used in the training process, they are used to evaluate the model
    
    the_pipe0 = Pipeline([
        ('scaler0', preprocessing.StandardScaler()),
        ('estimator0', SVR())
    ])

    kf0 = KFold(n_splits=5, shuffle=True, random_state=42)


    param_grid0 = {
    'estimator0__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'estimator0__C': [0.5, 1, 2],
    'estimator0__epsilon': [0.1, 0.2],
}



    grid_search0 = GridSearchCV(the_pipe0, param_grid0, cv=kf0, n_jobs=-2, verbose=2, scoring= 'r2') # leave one core for other operations

    grid_search0.fit(X_train0, y_train0)

    logger.info(f"Best param for our approach0: {grid_search0.best_params_}")
    logger.info(f"Best cross-validated R2 for our approach0: {grid_search0.best_score_}")

    best_model0 = grid_search0.best_estimator_
    y_pred0 = best_model0.predict(X_test0)

    r2 = r2_score(y_test0, y_pred0)
    logger.info(f"R2 score on test set for our approach0: {r2}")
    rmse = sqrt(mean_squared_error(y_test0, y_pred0))
    logger.info(f"RMSE on test set for our approach0: {rmse}")

    prediction_set0.loc[:,TARGET_NAME] = grid_search0.predict(prediction_set0.drop(columns=[TARGET_NAME]))

    prediction_set0[TARGET_NAME].to_csv('out\\approach0_highDim_SVR.csv', index=False)
    prediction_set0.to_csv('out\\approach0_highDim_SVR_all.csv', index=False)

    

    break

