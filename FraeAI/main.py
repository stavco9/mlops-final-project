# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# libraries importing
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# additional modules
import sys
import FreeAI as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #model_name = 'convAE'
    model_name = 'LGBM'
    #f = 1
    f = 2
    tree_split = 'f2'
    #tree_split = 'accuracy'

    if model_name == 'LGBM':
        test_y = arr = np.load('test_y.npy')
        test_pred = np.load('test_pred.npy')
        test_x = pd.read_csv('test_x.csv')
        train_x = pd.read_csv('train_x.csv')
    elif model_name == 'convAE':
        test_y = arr = np.load('y_test.npy')
        test_pred = np.load('y_pred.npy')
        test_x = pd.read_csv('test_x_convAE.csv')
    df2 = F.main_func(model_name+str(f)+'feautrs-'+tree_split, test_x, test_pred, test_y, number_of_features=f, metric=tree_split)



    '''
    #result = pd.concat([df1, df2], ignore_index=True).sort_values(by='acc', ascending=False)
    result = df2.sort_values(by=['feature','acc'], ascending=False)
    result1 = result.loc[result.groupby('feature')['acc'].idxmin()]
    result.to_csv('result.csv', index=False)  # Set index=False if you don't want to save row indices
    print('train percentage')
    #F.train_percentage(train_x, res)
    '''
