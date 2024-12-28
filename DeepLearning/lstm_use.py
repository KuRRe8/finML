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
YEAR_LIMIT = 8

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

imputed = pd.concat([origin_imputed, pd.DataFrame(target_imputed, columns=[TARGET_NAME])], axis=1)

grouped_by_gv = imputed.groupby('gvkey',group_keys=True)

for key, group in grouped_by_gv:
    if len(group) < YEAR_LIMIT:
        imputed = imputed[imputed['gvkey'] != key]

imputed.to_csv(os.path.join(pathprefix,'temp',f'grouped_with_{YEAR_LIMIT}plus.csv'), index=False)

grouped_by_gv_1 = imputed.groupby('gvkey', group_keys=True)

all_filled_data = []
first_100_test_groups = []
other_groups = []

for key, group in grouped_by_gv_1:
    group = group.set_index('fyear').sort_index()
    full_range = range(1988, 2025)
    group = group.reindex(full_range)
    group['mask4lstm'] = group['gvkey'].isnull()
    group['gvkey'] = group['gvkey'].ffill().bfill()
    group = group.ffill().bfill()
    group.fillna(0, inplace=True)
    all_filled_data.append(group)

first_100_test_groups = all_filled_data[:100]
other_groups = all_filled_data[100:]

all_filled_data = pd.concat(all_filled_data).reset_index().rename(columns={'index': 'fyear'})
all_filled_data.to_csv(os.path.join(pathprefix,'temp','filled_data.csv'), index=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm = lstm_def.myLSTM(input_size=all_filled_data.shape[1]-3, hidden_size=64, output_size=37, num_layers=1).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

train_data = []
train_target = []
#train_mask = []
for group in other_groups:
    train_data.append(group.drop(columns=['gvkey', 'mask4lstm']).values)
    train_target.append(group[TARGET_NAME].values)
    #train_mask.append(group['mask4lstm'].values)

train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_target = torch.tensor(train_target, dtype=torch.float32).to(device)
#train_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)

test_data = []
test_target = []
#test_mask = []
for group in first_100_test_groups:
    test_data.append(group.drop(columns=['gvkey', 'mask4lstm']).values)
    test_target.append(group[TARGET_NAME].values)
    #test_mask.append(group['mask4lstm'].values)

test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
test_target = torch.tensor(test_target, dtype=torch.float32).to(device)
#test_mask = torch.tensor(test_mask, dtype=torch.bool).to(device)

train_data, val_data, train_target, val_target = train_test_split(
    train_data, train_target, test_size=0.2, random_state=42)

for epoch in range(100):
    lstm.train()
    optimizer.zero_grad()
    output = lstm(train_data)
    loss = criterion(output, train_target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch} loss: {loss.item()}')

lstm.eval()
val_output = lstm(val_data)
val_loss = criterion(val_output, val_target)
print(f'Validation loss: {val_loss.item()}')

test_output = lstm(test_data)
test_loss = criterion(test_output, test_target)
print(f'Test loss: {test_loss.item()}')

test_output_cpu = test_output.cpu()
test_output_np = test_output_cpu.detach().numpy()
test_target_cpu = test_target.cpu()
test_target_np = test_target_cpu.detach().numpy()

print(f'Test RMSE: {sqrt(mean_squared_error(test_target_np, test_output_np))}')
print(f'Test R2: {r2_score(test_target_np, test_output_np)}')

logger.info('Finished.')
