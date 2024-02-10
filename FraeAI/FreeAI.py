# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# libraries importing
import pandas as pd
import matplotlib.pyplot as plt

# additional modules
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#import graphviz
#import dtreeviz.trees
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import itertools
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import plot_tree
from IPython.display import display


def CreateModels(test_data , accurate_prediction, number_of_features, max_models_number = float('inf')):
  #list of all the returned modules
  models_res = []
  categories_list = list(test_data)
  num_of_models = 0
  models_order =  list(itertools.combinations(categories_list, number_of_features))
  #for categories in itertools.combinations(set(categories_list), number_of_features):
  for categories in models_order:
    if num_of_models > max_models_number:
      break
    num_of_models += 1

    if number_of_features == 1:
      f_names = [categories[0]]
    else:
      f_names = [categories[0], categories[1]]

    #Featurs values
    X = test_data[f_names]
    #whether we predict correctly for this input
    y = accurate_prediction

    #create Decision tree
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    #plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    #plot_tree(model, filled=True, feature_names=f_names, node_ids=True)
    #plt.savefig(f'{categories[0]}' + '.png')

    #simple tree visualization
    '''
    data = export_graphviz(model, feature_names=f_names,
                            filled=True, rounded=False, node_ids=True, precision=True)
    graph = graphviz.Source(data)
    if number_of_features == 1:
      graph.render(f'{categories[0]}')
    else:
      graph.render(f'{categories[0]}-{categories[1]}')
      #add another type of visualization
      viz_model = dtreeviz.model(model,
                      X,
                      y.astype(bool),
                      feature_names= f_names,
                      target_name= "accuracy_bool",
                      class_names=["True", "False"])
      v = viz_model.view()
      v.show()                 # pop up window
      v.save('trees/'+f'{categories[0]}-{categories[1]}'+'.svg')
    '''
    models_res.append(model)

  return models_res, models_order

def saveRelevantInfo2_no_acc(model,test_pred1, true_pred, data, original_acc, acc_deg, min_sample_number, max_impurity, number_of_features = 1, calcMetric = False):
  res = []
  test_pred= np.squeeze(test_pred1)

  m = model
  tree = m.tree_
  thresholds = [sorted(tree.threshold[tree.feature==0])]
  if number_of_features == 2:
      thresholds.append(sorted(tree.threshold[tree.feature==1]))
  leafs_indices = set(m.apply(data))
  leaf_list = m.apply(data)
  min__leaf = float('inf')
  min_leaf_perf = float('inf')
  for leaf_idx in leafs_indices:
    if tree.n_node_samples[leaf_idx] > min_sample_number and tree.impurity[leaf_idx] < max_impurity:
      indices = tree.value[leaf_idx]
      acc = indices[0][1] / sum(indices[0])
      true_prediction = indices[0][1]
      false_prediction = indices[0][0]
      if (True):
          leaf = leaf_list == leaf_idx
          samples_in_leaf = np.where(leaf)[0]
          data_in_leaf = data.values[samples_in_leaf]
          if data_in_leaf.size == 0:
              print("Leaf node is empty.")
          else:
              # Analyze the range of values for each feature
              minVal = [float('-inf') for x in range(number_of_features)]
              maxVal = [float('inf') for x in range(number_of_features)]
              for f in range(number_of_features):
                for th in range(len(thresholds[f])):
                  if thresholds[f][th]!=-2:
                    if data_in_leaf[0][f] < thresholds[f][th]:
                      #res.append((leaf_idx, acc,data.columns[0], float('-inf'), thresholdsA[th]))
                        maxVal[f] = thresholds[f][th]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: smaller than" +  str(thresholds[f][th]))
                        break
                    elif data_in_leaf[0][f] > thresholds[f][th]:
                      if th == len(thresholds[f])-1:
                        #res.append((leaf_idx, acc,data.columns[0], thresholdsA[th], float('inf')))
                        minVal[f] = thresholds[f][th]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: larger than" +  str(thresholds[f][th]))
                        break
                      elif data_in_leaf[0][f] < thresholds[f][th+1]:
                        #res.append((leaf_idx, acc,data.columns[0], thresholdsA[th], thresholdsA[th+1]))
                        minVal[f] = thresholds[f][th]
                        maxVal[f] = thresholds[f][th+1]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: " +  str(thresholds[f][th]) + " , " + str(thresholds[f][th+1]))
                        break
              feature_ranges = [(data_in_leaf[:, i].min(), data_in_leaf[:, i].max()) for i in range(data_in_leaf.shape[1])]

              orig_recall = recall_score(true_pred, test_pred)
              orig_precision = precision_score(true_pred, test_pred)
              original_f2 = fbeta_score(true_pred, test_pred, average='binary', beta=2)
              max_index = data.index.max()
              all_indexes = set(range(max_index + 1))
              if number_of_features == 1:
                correct_indices = data[(data[data.columns[0]].between(feature_ranges[0][0], feature_ranges[0][1]))].index
                non_slice_indices = sorted(all_indexes - set(correct_indices))
                slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
                slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
                slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
                all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary',beta=2)
                all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                f2_improvment = all_f2 - original_f2
                res.append((leaf_idx, slice_f2,f2_improvment,slice_precision, slice_recall, len(correct_indices),true_prediction,false_prediction,data.columns[0], minVal[0],maxVal[0]))
              else:
                correct_indices = data[(data[data.columns[0]].between(feature_ranges[0][0], feature_ranges[0][1])) & (data[data.columns[1]].between(feature_ranges[1][0], feature_ranges[1][1]))].index
                non_slice_indices = sorted(all_indexes - set(correct_indices))
                slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
                slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
                slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
                all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary',beta=2)
                all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                f2_improvment = all_f2 - original_f2
                res.append((leaf_idx, slice_f2,f2_improvment,slice_precision, slice_recall, len(correct_indices),true_prediction,false_prediction,data.columns[0], minVal[0],maxVal[0],data.columns[1], minVal[1],maxVal[1]))
              if slice_f2 < min_leaf_perf:
                min_leaf = leaf_idx
                min_leaf_perf = slice_f2
  '''
  f2_improvment = 0
  if calcMetric == True:
    leaf = leaf_list == min_leaf
    samples_in_leaf = np.where(leaf)[0]
    data_in_leaf = data.values[samples_in_leaf]
    feature_ranges = [(data_in_leaf[:, i].min(), data_in_leaf[:, i].max()) for i in range(data_in_leaf.shape[1])]
    correct_indices = data[(data[data.columns[0]].between(feature_ranges[0][0], feature_ranges[0][1]))].index
    max_index = data.index.max()
    all_indexes = set(range(max_index + 1))
    non_slice_indices = sorted(all_indexes - set(correct_indices))
    slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
    slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
    slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
    all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary', beta=2)
    all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
    all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
    orig_recall = recall_score(true_pred, test_pred)
    orig_precision = precision_score(true_pred, test_pred)
    original_f2 = fbeta_score(true_pred, test_pred, average='binary', beta=2)

    #now calculate the metric
    f2_improvment = all_f2 - original_f2
  '''''
  return res



def main_func_max(test_data, pred_test, y_test, number_of_features = 1, max_models_number = float('inf'), metric = 'TP'):
  if metric == 'accuracy':
    relevant_predictions = (np.array(pred_test).flatten() ==  np.array(y_test))
    test_x_ = test_data
    result, order = CreateModels(test_x_ , relevant_predictions, number_of_features, max_models_number)
  elif metric == 'TP':
    #keep only predictions for anomalies
    relevant_predictions = np.array(pred_test).flatten()[np.array(y_test)!=0]
    test_x_ = test_data[np.array(y_test)!=0]
    result, order = CreateModels(test_x_ , relevant_predictions, number_of_features, max_models_number)

  res = []
  worst_model_idx = None
  worst_model_imp = float('-inf')
  for f in range(min(max_models_number,len(order))):
    model = result[f]## model returned from the classifier
    original_acc =  sum(relevant_predictions) / len(relevant_predictions)
    original_f2 = fbeta_score(y_test, pred_test, average='binary', beta=2)
    acc_deg = 0.05
    min_sample_number = 0.1* len(y_test)
    max_impurity = 1
    if number_of_features == 1:
      test_features = test_x_[[order[f][0]]]
    else:
      test_features = test_x_[[order[f][0],order[f][1]]]
    new_res = saveRelevantInfo2_no_acc(model,pred_test,y_test,test_features, original_acc,acc_deg,min_sample_number,max_impurity,number_of_features, True)
    res.extend(new_res)

  sorted_items = sorted(res, key=lambda x: x[2], reverse=True)

  if number_of_features == 1:
    df = pd.DataFrame(sorted_items, columns=['leaf id', 'acc','acc imp','precision', 'recall', 'size', 'correct_pred','false_pred','feature',  'min val1', 'max val1'])
  else:
    df = pd.DataFrame(sorted_items, columns=['leaf id', 'acc','acc imp', 'precision', 'recall', 'size','correct_pred','false_pred','feature',  'min val1', 'max val1', 'feature_2',  'min val2', 'max val2'])

  result_2 = df.sort_values(by=['feature', 'acc imp'], ascending=False)
  result1 = result_2.loc[result_2.groupby('feature')['acc imp'].idxmax()]
  df_sorted = result1.sort_values(by='acc imp', ascending=False)
  print("\nUsing IPython.display.display():")
  display(df_sorted)

  # Plotting the bar chart
  plt.figure(figsize=(10, 6))
  plt.bar(df_sorted['feature'], df_sorted['acc imp'])
  plt.xlabel('Feature')
  plt.ylabel('Potential accuracy Improvement')
  plt.title('Potential accuracy improvement for identified slice')
  plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()

  worst_feature = df_sorted.iloc[0]['feature']
  for index, tup in enumerate(order):
    if tup[0] == worst_feature:
      model = result[index]  ## model returned from the classifier
      original_acc = sum(relevant_predictions) / len(relevant_predictions)
      min_sample_number = 0#0.1 * len(y_test)
      max_impurity = 1
      if number_of_features == 1:
        test_features = test_x_[[order[index][0]]]
      else:
        test_features = test_x_[[order[index][0], order[index][1]]]
      new_res = saveRelevantInfo2_no_acc(model, pred_test, y_test, test_features, original_acc, 0,min_sample_number, max_impurity, number_of_features, True)
      if number_of_features == 1:
        df = pd.DataFrame(new_res,
                          columns=['leaf id', 'acc', 'acc imp', 'precision', 'recall', 'size', 'correct_pred',
                                   'false_pred', 'feature', 'min val1', 'max val1'])
      else:
        df = pd.DataFrame(new_res,
                          columns=['leaf id', 'acc', 'acc imp', 'precision', 'recall', 'size', 'correct_pred',
                                   'false_pred', 'feature', 'min val1', 'max val1', 'feature_2', 'min val2',
                                   'max val2'])
      plt.figure(figsize=(10, 6))
      plt.bar(df_sorted['leaf id'], df_sorted['acc imp'], tick_label=df_sorted['leaf id'])
      plt.xlabel('Leaf ID')
      plt.ylabel('Potential accuracy Improvement')
      plt.title('Potential accuracy improvement for identified slice')
      plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
      plt.tight_layout()  # Adjust layout to prevent clipping of labels
      plt.show()
      break
  '''
  # Extract names and numbers from sorted data
  names = [x[0] for x in sorted_metric]
  mtrics = [x[1] for x in sorted_metric]
  # Create bar plot
  plt.bar(names, mtrics)
  plt.xlabel('Names')
  plt.ylabel('Numbers')
  plt.title('Bar Plot')
  plt.xticks(rotation=90)

  plt.show()
  '''
  return result1


#def calc_metric_per_feature()
def keepFeature(df, f1 = None, f2 = None):
  if f2 == None:
    return df[(df['feature'] == f1) & (df['feature_2'].isna())]
  else:
    return df[(df['feature'] == f1) & (df['feature_2'] == f2)]

def calcMetric(df, f1=None, f2=None):
  feature_res =  keepFeature(df,f1,f2)
  best_acc = feature_res['acc'].max()
  max_acc_row = feature_res[feature_res['acc'] == feature_res['acc'].max()]
  worst_acc = feature_res['acc'].min()
  min_acc_row = feature_res[feature_res['acc'] == feature_res['acc'].min()]
  print((best_acc,worst_acc))
  return(pd.concat([max_acc_row, min_acc_row]))


def train_percentage(test_x, df):
  for i in range(2):
    first_row_values = df.iloc[i]
    feature = first_row_values['feature']
    min_val1 = first_row_values['min val1']
    max_val1 = first_row_values['max val1']
    feature_2 = first_row_values['feature_2']
    min_val2 = first_row_values['min val2']
    max_val2 = first_row_values['max val2']
    filtered_data = test_x[(test_x[feature] >= min_val1) & (test_x[feature] <= max_val1)]
    # Calculate the number of rows within the range
    count_within_range = len(filtered_data)
    # Calculate the total number of rows in the DataFrame
    total_count = len(test_x)
    # Calculate the percentage of rows within the range
    percentage = (count_within_range / total_count) * 100
    if i == 0:
      print("max slide percentage")
    else:
      print("min slide percentage")
    print(percentage)
