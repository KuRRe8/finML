import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

pathprefix = os.path.dirname(os.path.abspath(__file__))

file_list = [
    os.path.join(pathprefix, 'out', 'linear_bayesian.csv'),
    os.path.join(pathprefix, 'out', 'linear_lasso.csv'),
    os.path.join(pathprefix, 'out', 'linear_OLS.csv'),
    os.path.join(pathprefix, 'out', 'linear_ridge.csv'),
    os.path.join(pathprefix, 'out', 'neural_MLP.csv'),
    os.path.join(pathprefix, 'out', 'tree_RF.csv'),
    os.path.join(pathprefix, 'out', 'XGBRegressor.csv'),
    os.path.join(pathprefix, 'out', 'DeepLearning', 'out', 'tabresnet.csv')
]

results = {}
for filename in file_list:
    df = pd.read_csv(filename)
    df = df.sort_values('xrd')
    df['group'] = pd.qcut(df['xrd'], q=10, labels=False)
    mean_q = df.groupby(['group'])['q'].mean().values
    basename = os.path.basename(filename)
    fn_wo_ext = os.path.splitext(basename)[0]
    results[fn_wo_ext] = mean_q
    plt.figure()
    for fn, q_values in results.items():
        plt.plot(range(1, 11), q_values, label=fn)

    plt.xticks(range(1, 11))
    plt.legend()
    plt.xlabel('Decile')
    plt.ylabel('Mean Q')
    plt.title('Decile-wise Average of Q')
    #plt.yscale('log')
    plt.savefig(os.path.join(pathprefix, 'decilewise_average_of_q.png'))
