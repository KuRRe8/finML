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



import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_widedeep.models import TabResnet
from pytorch_widedeep.preprocessing import TabPreprocessor



"""
Step 0: Set param
"""
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


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
filehandler = logging.FileHandler('out\\tabresnet.log')
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


myImputer = SimpleImputer(missing_values = pd.NA, strategy='mean')
X_train = pd.DataFrame(myImputer.fit_transform(X_train), columns= X_train.columns)
X_test = pd.DataFrame(myImputer.transform(X_test), columns= X_test.columns)
prediction_set = prediction_set.drop(columns=[TARGET_NAME]) # drop the empty target column
prediction_set = pd.DataFrame(myImputer.transform(prediction_set), columns= prediction_set.columns)
#myscaler = preprocessing.StandardScaler()

#use tabpreprocessor instead of myImputer and myscaler
cat_embed_cols = ['gvkey']
continuous_cols=X_train.columns.tolist()[1:]
tab_preprocessor = TabPreprocessor(cat_embed_cols=['gvkey'], continuous_cols=X_train.columns.tolist()[1:], cols_to_scale='all')
X_train_processed = tab_preprocessor.fit_transform(X_train)
X_test_processed = tab_preprocessor.transform(X_test)
prediction_set_processed = tab_preprocessor.transform(prediction_set)

tab_resnet = TabResnet(column_idx=tab_preprocessor.column_idx, 
                       cat_embed_input=tab_preprocessor.cat_embed_input, 
                       continuous_cols=continuous_cols,
                       blocks_dims=[10,1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tab_resnet.to(device)

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
prediction_set_processed_tensor = torch.tensor(prediction_set_processed, dtype=torch.float32).to(device)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(tab_resnet.parameters(), lr=0.01)

#no KFold, no cross validation
num_epochs = 50000
tab_resnet.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = tab_resnet(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

tab_resnet.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    predictions = tab_resnet(X_test_tensor)
    loss = criterion(predictions, y_test_tensor)
    logger.info(f'Loss on test set: {loss.item()}')

    y_test_tensor = y_test_tensor.cpu().numpy()
    predictions = predictions.cpu().numpy()
    rmse = sqrt(mean_squared_error(y_test_tensor, predictions))
    r2 = r2_score(y_test_tensor, predictions)
    logger.info(f'RMSE: {rmse}, R2: {r2}')
    
    prediction_set.loc[:,TARGET_NAME] = tab_resnet(prediction_set_processed_tensor).cpu().numpy()

    prediction_set[TARGET_NAME].to_csv('out\\tabresnet.csv', index=False)
