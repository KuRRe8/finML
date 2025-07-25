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
print(f"Hostname: {os.getenv('HOSTNAME')}")
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

num_epochs = 100000
patience = 600  # 早停耐心，表示在验证集上表现没有提升的epochs数
min_delta = 0.0000001  # 最小的改善幅度

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
with open(os.path.join(pathprefix,'..','datapath'), 'r') as f:
    origin_path = f.readline().strip()
METHODNAME = 'tabresnet'

logger = logging.getLogger()
filehandler = logging.FileHandler(os.path.join(pathprefix,'out', f'{METHODNAME}.log'))
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(filehandler)
logger.setLevel(logging.INFO)

logger.info('Started.')

"""
Step 1: Data Cleaning
"""

if origin_path.endswith('.csv'):
    origin = pd.read_csv(origin_path,low_memory=False)
elif origin_path.endswith('.dta'):
    origin = pd.read_stata(origin_path)
logger.info(f"Data loaded from {origin_path}")
origin = origin.drop(columns=ALWAYS_DROP_LIST, errors='ignore')
preserve_cols:pd.DataFrame = origin[PRESERVE_LIST]
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
#hasher = sklearn.feature_extraction.FeatureHasher(n_features=2**18, input_type='string') #use 256K hash space for 37.4K distinct gvkey
s_gvkey = s_gvkey.astype('int64')
origin['gvkey'] = s_gvkey
# DO NOT need to hash gvkey, we will process it in TabPreprocessor

# datetime641: datadate
base_date = pd.Timestamp('1987-01-01')
if 'datadate' in origin.columns:
    origin['datadate'] = (pd.to_datetime(origin['datadate']) - base_date).dt.days

if _DEBUG_FALSE:
    origin.to_csv(os.path.join(pathprefix,'temp','s3converttype.csv'), index=False)

# all other objects/strings dropped. tic, cusip, conm, cik, add2, weburl, etc.
droplist = [col for col in origin.columns if origin[col].dtype == 'object']
origin.drop(droplist, axis=1, inplace=True)

if _DEBUG:
    origin.to_csv(os.path.join(pathprefix,'temp','s4beforesplit.csv'), index=False)
    origin.describe().to_csv(os.path.join(pathprefix,'temp','s4beforesplit_desc.csv'))
common_columns = origin.columns.intersection(preserve_cols.columns)
origin = origin.drop(columns=common_columns)
origin = pd.concat([origin, preserve_cols], axis=1)

if TARGET_NAME not in origin.columns:
    origin = pd.concat([origin, target_col], axis=1) # always preserve target column

# devide prediction set and train_test set as early as possible

prediction_set = origin[origin[TARGET_NAME].isnull()]
train_test_set = origin[origin[TARGET_NAME].notnull()]

'''
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

'''
# standardize the data, process 400k samples together
scaler = preprocessing.StandardScaler().fit(imputed.to_numpy())

step1_final = pd.DataFrame(scaler.transform(imputed.to_numpy()), columns=imputed.columns)

if NEED_PCA:
    NotImplementedError('PCA not implemented yet')
else:
    pass
'''

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logger.debug):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
                       blocks_dims=[64,16,1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'use {device}')
tab_resnet.to(device)

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
prediction_set_processed_tensor = torch.tensor(prediction_set_processed, dtype=torch.float32).to(device)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(tab_resnet.parameters(), lr=0.01)

#no KFold, no cross validation
es = EarlyStopping(patience=patience, delta=min_delta)
tab_resnet.train()
for epoch in range(num_epochs):
    
    tab_resnet.train()
    optimizer.zero_grad()
    outputs = tab_resnet(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()


    tab_resnet.eval()
    with torch.no_grad():
        X_test_tensor_inner = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
        y_test_tensor_inner = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
        predictions_inner = tab_resnet(X_test_tensor_inner)
        loss = criterion(predictions_inner,y_test_tensor_inner)
        es(loss,tab_resnet)
    
    if es.early_stop:
        print('early stopped, in epoch' + str(epoch))
        break

    if epoch % 500 == 0:
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
    ajusted_r2 = 1 - ((1 - r2) * (X_test_tensor.shape[0] - 1) / (X_test_tensor.shape[0] - X_test_tensor.shape[1] - 1))
    logger.info(f'Adjusted R2: {ajusted_r2}')
    prediction_set.loc[:,TARGET_NAME] = tab_resnet(prediction_set_processed_tensor).cpu().numpy()

    prediction_set.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}.csv'), index=False)

    origin = pd.concat([origin, s_gvkey], axis=1)
    origin.loc[origin[TARGET_NAME].isnull(), TARGET_NAME] = prediction_set[TARGET_NAME]
    origin.to_csv(os.path.join(pathprefix,'out',f'{METHODNAME}_full.csv'), index=False)