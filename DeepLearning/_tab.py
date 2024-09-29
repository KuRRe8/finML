import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_widedeep.models import TabResnet
from pytorch_widedeep.preprocessing import TabPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


toy_annfunda = {
    'gvkey': [1000, 1001, 1002, 1003, 1004],# categorical
    'gind': [10, 15, 10, 20, 15],# numerical
    'gsector': [1, 1, 2, 2, 1],# numerical
    'ggroup': [1010, 1510, 1020, 2010, 1510],#numerical
    'fyear': [2010, 2010, 2010, 2010, 2010],#numerical
}

toy_annfunda_target = [0.1, 0.2, 0.3, 0.4, 0.5]

prediction_set = pd.DataFrame(toy_annfunda)

dataset = pd.DataFrame(toy_annfunda)
dataset['xrd'] = toy_annfunda_target

# Split the dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('xrd', axis=1), dataset['xrd'], test_size=0.2, random_state=0)

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

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(tab_resnet.parameters(), lr=0.01)

#no KFold, no cross validation
num_epochs = 7
tab_resnet.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = tab_resnet(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

tab_resnet.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    predictions = tab_resnet(X_test_tensor)
    loss = criterion(predictions, y_test_tensor)
    print(f'Loss on test set: {loss.item()}')

    y_test_tensor = y_test_tensor.cpu().numpy()
    predictions = predictions.cpu().numpy()
    rmse = sqrt(mean_squared_error(y_test_tensor, predictions))
    r2 = r2_score(y_test_tensor, predictions)
    print(f'RMSE: {rmse}, R2: {r2}')
    
    prediction_set.loc[:,'xrd'] = tab_resnet(prediction_set_processed).cpu().numpy()
