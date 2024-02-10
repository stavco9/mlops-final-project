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
#import graphviz
#import dtreeviz.trees
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import itertools
from sklearn.metrics import accuracy_score
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_y = arr = np.load('test_y.npy')
    test_pred = np.load('test_pred.npy')
    test_x = pd.read_csv('test_x.csv')
    train_x = pd.read_csv('train_x.csv')
    #df1 = F.main_func_max(test_x, test_pred, test_y, number_of_features=2, metric='accuracy')
    df2 = F.main_func_max(test_x, test_pred, test_y, number_of_features=1, metric='accuracy')
    #result = pd.concat([df1, df2], ignore_index=True).sort_values(by='acc', ascending=False)
    result = df2.sort_values(by=['feature','acc'], ascending=False)
    result1 = result.loc[result.groupby('feature')['acc'].idxmin()]

    result.to_csv('result.csv', index=False)  # Set index=False if you don't want to save row indices
    pd.set_option('display.max_columns', None)
    #res = F.calcMetric(result, 'A1_max')

    print('train percentage')
    #F.train_percentage(train_x, res)

    print('test percentage')
    #F.train_percentage(test_x, res)
    #print("\nUsing IPython.display.display():")
    #display(result1)
    #print(res)
